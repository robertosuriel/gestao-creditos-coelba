import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
import json
import plotly.express as px
import plotly.graph_objects as go

# =================================================================
# 1. CONFIGURAÇÕES E SEGURANÇA
# =================================================================
st.set_page_config(page_title="Blue Solutions - Gestão Energética", layout="wide")

# Tenta pegar dos Segredos, senão usa string direta (Fallback para testes locais)
DB_CONFIG = st.secrets["database"]["url"]
MINHA_API_KEY = st.secrets["gemini"]["api_key"]

# Conexão Banco
engine = create_engine(DB_CONFIG)

MESES_MAP = {
    'jan': 'Jan', 'fev': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 
    'mai': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 
    'set': 'Sep', 'out': 'Oct', 'nov': 'Nov', 'dez': 'Dec'
}

# =================================================================
# 2. FUNÇÕES DE BANCO DE DADOS
# =================================================================

def listar_relatorios_salvos():
    query = text("SELECT id_auditoria, geradora_nome, periodo_apuracao FROM audit_fatura_cabecalho ORDER BY id_auditoria DESC")
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def verificar_relatorio_completo(geradora_nome, periodo):
    query = text("""
        SELECT c.id_auditoria, COUNT(r.id) as qtd_rateio 
        FROM audit_fatura_cabecalho c
        LEFT JOIN audit_fatura_analise_rateio r ON c.id_auditoria = r.id_auditoria
        WHERE c.geradora_nome = :gn AND c.periodo_apuracao = :per
        GROUP BY c.id_auditoria
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"gn": geradora_nome, "per": periodo}).fetchone()
        if result and result[1] > 0:
            return int(result[0])
        return None

def carregar_dados_banco(id_auditoria):
    id_auditoria = int(id_auditoria)
    query_hist = text("""
        SELECT situacao_mensal, codigo_cliente, consumo_kwh AS consumo, 
               injetado_kwh AS injetado, data_referencia AS data_ref 
        FROM audit_fatura_historico WHERE id_auditoria = :id
    """)
    query_rateio = text("""
        SELECT codigo_cliente, saldo_acumulado_kwh AS saldo_acumulado, 
               rateio_atual_decimal * 100 AS percentual_compensacao,
               media_consumo_6m, credito_ideal_kwh, meses_autonomia,
               rateio_ideal_decimal * 100 as rateio_ideal_pct,
               proximo_rateio_decimal * 100 as proximo_rateio_pct,
               observacao_gestao
        FROM audit_fatura_analise_rateio WHERE id_auditoria = :id
    """)
    hist = pd.read_sql(query_hist, engine, params={"id": id_auditoria})
    rateio = pd.read_sql(query_rateio, engine, params={"id": id_auditoria})
    return hist, rateio

def salvar_processamento(meta, df_hist, df_rateio):
    with engine.begin() as conn:
        media_med = float(df_hist['injetado'].tail(6).mean()) if not df_hist['injetado'].empty else 0.0
        res = conn.execute(text("""
            INSERT INTO audit_fatura_cabecalho (geradora_nome, periodo_apuracao, media_geracao_mensal)
            VALUES (:gn, :per, :med) RETURNING id_auditoria
        """), {"gn": meta.get('geradora_nome', 'Desconhecido'), "per": meta.get('periodo', 'N/A'), "med": media_med})
        id_auditoria = res.fetchone()[0]

        # Histórico
        df_hist_sql = df_hist.copy()
        df_hist_sql['id_auditoria'] = id_auditoria
        map_hist = {'consumo': 'consumo_kwh', 'injetado': 'injetado_kwh', 'data_ref': 'data_referencia'}
        if 'data_ref' not in df_hist_sql.columns: df_hist_sql['data_ref'] = df_hist_sql['situacao_mensal'].apply(parse_date)
        df_hist_sql = df_hist_sql.rename(columns=map_hist)
        cols_hist = ['id_auditoria', 'situacao_mensal', 'codigo_cliente', 'consumo_kwh', 'injetado_kwh', 'data_referencia']
        df_hist_sql[cols_hist].to_sql('audit_fatura_historico', conn, if_exists='append', index=False)

        # Rateio
        df_rateio_sql = df_rateio.copy()
        df_rateio_sql['id_auditoria'] = id_auditoria
        map_rateio = {'saldo_acumulado': 'saldo_acumulado_kwh', 'Média Consumo': 'media_consumo_6m',
                      'Crédito Ideal': 'credito_ideal_kwh', 'Meses de Crédito': 'meses_autonomia', 'Observações': 'observacao_gestao'}
        df_rateio_sql = df_rateio_sql.rename(columns=map_rateio)
        
        # Converte para decimal (banco armazena 0.07 para 7%)
        if 'percentual_compensacao' in df_rateio_sql.columns: df_rateio_sql['rateio_atual_decimal'] = df_rateio_sql['percentual_compensacao'] / 100
        if 'Rateio Ideal %' in df_rateio_sql.columns: df_rateio_sql['rateio_ideal_decimal'] = df_rateio_sql['Rateio Ideal %'] / 100
        if 'Próximo Rateio %' in df_rateio_sql.columns: df_rateio_sql['proximo_rateio_decimal'] = df_rateio_sql['Próximo Rateio %'] / 100
        
        cols_rateio = ['id_auditoria', 'codigo_cliente', 'saldo_acumulado_kwh', 'media_consumo_6m',
                       'credito_ideal_kwh', 'meses_autonomia', 'rateio_atual_decimal', 'rateio_ideal_decimal', 
                       'proximo_rateio_decimal', 'observacao_gestao']
        cols_finais = [c for c in cols_rateio if c in df_rateio_sql.columns]
        df_rateio_sql[cols_finais].to_sql('audit_fatura_analise_rateio', conn, if_exists='append', index=False)
    return id_auditoria

# =================================================================
# 3. INTEGRAÇÃO COM GEMINI (IA)
# =================================================================

@st.cache_data(show_spinner=False)
def process_with_gemini(file_bytes, api_key):
    client = genai.Client(api_key=api_key)
    prompt = """
    Analise o PDF e extraia os dados estritamente no formato JSON.
    ESTRUTURA:
    {
      "metadata": {"geradora_nome": "string", "periodo": "string"},
      "tabela_13_meses": [{"situacao_mensal": "mes/ano", "codigo_cliente": "string", "consumo": 0.0, "injetado": 0.0}],
      "resumo_saldo": [{"codigo_cliente": "string", "percentual_compensacao": 0.0, "saldo_acumulado": 0.0}]
    }
    """
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

@st.cache_data(show_spinner=False)
def sugerir_proximo_rateio(dados_json, id_geradora, api_key):
    client = genai.Client(api_key=api_key)
    prompt_ia = f"""
    Analise estes dados: {dados_json}
    Sugira o 'Próximo Rateio %' para equilibrar créditos da usina.
    
    REGRAS DE OURO (META: 2 MESES DE AUTONOMIA):
    1. Unidade Geradora ({id_geradora}): Rateio deve ser 0.0.
    2. Se Meses de Crédito > 2.5: Reduza o rateio (Crédito Alto).
    3. Se Meses de Crédito < 1.7: Aumente o rateio (Crédito Baixo).
    4. Tente manter a soma próxima de 100%.
    
    Retorne JSON: {{"codigo_cliente": valor_float}}
    """
    resp = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt_ia)
    try:
        sugestoes = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
    except:
        return {} 

    # Normalização Matemática
    if id_geradora in sugestoes: sugestoes[id_geradora] = 0.0
    soma = sum(sugestoes.values())
    if soma > 0:
        fator = 100.0 / soma
        for k in sugestoes: sugestoes[k] = round(sugestoes[k] * fator, 2)
        
        # Ajuste fino
        nova_soma = sum(sugestoes.values())
        diff = 100.0 - nova_soma
        if abs(diff) > 0.0001:
            clientes_validos = {k: v for k, v in sugestoes.items() if k != id_geradora}
            if clientes_validos:
                maior = max(clientes_validos, key=clientes_validos.get)
                sugestoes[maior] = round(sugestoes[maior] + diff, 2)
    return sugestoes

def clean_val(val):
    if val is None or val == "": return 0.0
    if isinstance(val, (int, float)): return float(val)
    return float(str(val).replace('.', '').replace(',', '.'))

def parse_date(date_str):
    if not isinstance(date_str, str): return pd.NaT
    try:
        parts = date_str.split('/')
        mes_en = MESES_MAP.get(parts[0].lower(), parts[0])
        return pd.to_datetime(f"{mes_en}/{parts[1]}", format='%b/%Y')
    except:
        return pd.NaT

# =================================================================
# 4. LÓGICA PRINCIPAL (STREAMLIT)
# =================================================================

st.title("📊 Painel de Gestão Energética - Blue Solutions")

# Sidebar: Histórico
df_relatorios = listar_relatorios_salvos()
opcao_selecionada = st.sidebar.selectbox("Carregar relatório existente:", options=["Novo Upload"] + df_relatorios['geradora_nome'].unique().tolist())

df_historico_raw, df_rateio_final, meta_dados = None, None, {}
usar_dados_banco = False

# Carregamento de Dados
if opcao_selecionada != "Novo Upload":
    row = df_relatorios[df_relatorios['geradora_nome'] == opcao_selecionada].iloc[0]
    id_auditoria = int(row['id_auditoria'])
    with st.spinner("Carregando do Banco AWS..."):
        df_historico_raw, df_rateio_final = carregar_dados_banco(id_auditoria)
        meta_dados = {'geradora_nome': row['geradora_nome'], 'periodo': row['periodo_apuracao']}
        usar_dados_banco = True
        st.success(f"✅ {row['periodo_apuracao']} carregado!")
else:
    uploaded_files = st.file_uploader("Arraste seus PDFs aqui", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_file_name = st.sidebar.selectbox("Arquivo Atual:", file_names)
        selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
        
        with st.spinner("Processando IA..."):
            dados_ia = process_with_gemini(selected_file.getvalue(), MINHA_API_KEY)
            meta_dados = dados_ia.get('metadata', {})
            id_existente = verificar_relatorio_completo(meta_dados.get('geradora_nome'), meta_dados.get('periodo'))
            
            if id_existente:
                st.info("🔄 Encontrado no banco. Carregando...")
                df_historico_raw, df_rateio_final = carregar_dados_banco(id_existente)
                usar_dados_banco = True
            else:
                st.warning("⚡ Novo processamento IA iniciado...")
                df_historico_raw = pd.DataFrame(dados_ia['tabela_13_meses'])
                df_saldo_raw = pd.DataFrame(dados_ia['resumo_saldo'])
                for col in ['consumo', 'injetado']: df_historico_raw[col] = df_historico_raw[col].apply(clean_val)
                for col in ['saldo_acumulado', 'percentual_compensacao']: df_saldo_raw[col] = df_saldo_raw[col].apply(clean_val)
                df_rateio_final = df_saldo_raw.copy()

# Renderização do Dashboard
if df_historico_raw is not None and df_rateio_final is not None:
    
    # Processamento Gráficos
    df_historico_raw['data_ref'] = df_historico_raw['situacao_mensal'].apply(parse_date)
    df_historico_raw = df_historico_raw.sort_values('data_ref')
    id_geradora = df_historico_raw.groupby('codigo_cliente')['injetado'].sum().idxmax()
    df_geradora = df_historico_raw[df_historico_raw['codigo_cliente'] == id_geradora].copy()
    df_beneficiarias = df_historico_raw[df_historico_raw['codigo_cliente'] != id_geradora].copy()
    media_geracao = df_geradora['injetado'].tail(6).mean()

    # Cálculos Matemáticos (se não veio do banco)
    if not usar_dados_banco:
        avg_consumo = df_beneficiarias.groupby('codigo_cliente')['consumo'].apply(lambda x: x.tail(6).mean()).reset_index()
        df_rateio_final = df_rateio_final.merge(avg_consumo, on='codigo_cliente', how='left')
        df_rateio_final['Média Consumo'] = df_rateio_final['consumo'].fillna(0)
        df_rateio_final['Crédito Ideal'] = df_rateio_final['Média Consumo'] * 2
        df_rateio_final['Meses de Crédito'] = (df_rateio_final['saldo_acumulado'] / df_rateio_final['Média Consumo']).replace([float('inf')], 0).fillna(0)
        df_rateio_final['Rateio Ideal %'] = (df_rateio_final['Média Consumo'] / media_geracao * 100)
        
        # IA Sugestão
        resumo_json = df_rateio_final[['codigo_cliente', 'Rateio Ideal %', 'percentual_compensacao', 'Meses de Crédito']].to_json()
        try:
            sugestoes = sugerir_proximo_rateio(resumo_json, id_geradora, MINHA_API_KEY)
            df_rateio_final['Próximo Rateio %'] = df_rateio_final['codigo_cliente'].map(sugestoes).fillna(df_rateio_final['Rateio Ideal %'])
        except:
            df_rateio_final['Próximo Rateio %'] = df_rateio_final['Rateio Ideal %']

        # Definição de Observações (NOVAS REGRAS)
        def definir_obs(row):
            if row['Meses de Crédito'] > 2.5: return "Crédito Alto: Sugerido reduzir rateio"
            if row['Meses de Crédito'] < 1.7 and row['codigo_cliente'] != id_geradora: return "Crédito Baixo: Sugerido aumentar rateio"
            return "Equilibrado"
        
        df_rateio_final['Observações'] = df_rateio_final.apply(definir_obs, axis=1)
        
        # TRAVA DE SEGURANÇA: Zera a geradora
        df_rateio_final.loc[df_rateio_final['codigo_cliente'] == id_geradora, 'Próximo Rateio %'] = 0.0
        
        salvar_processamento(meta_dados, df_historico_raw, df_rateio_final)

    # Header
    st.info(f"Relatório: **{meta_dados.get('periodo')}** | Geradora: **{meta_dados.get('geradora_nome')}** | Média Geração: **{media_geracao:,.2f} kWh**")

    # Gráficos
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.line(df_geradora, x='situacao_mensal', y='injetado', markers=True, color_discrete_sequence=['#2E7D32'], title="Histórico Geração"), use_container_width=True)
    with c2: st.plotly_chart(px.line(df_beneficiarias, x='situacao_mensal', y='consumo', color='codigo_cliente', markers=True, title="Histórico Consumo"), use_container_width=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Histórico", "💰 Saldos", "🎯 Análise de Rateio"])

    with tab1:
        st.dataframe(df_historico_raw[['situacao_mensal', 'codigo_cliente', 'consumo', 'injetado']], use_container_width=True)

    with tab2:
        df_view_saldo = df_rateio_final[['codigo_cliente', 'saldo_acumulado', 'percentual_compensacao']].copy()
        df_view_saldo['saldo_acumulado'] = df_view_saldo['saldo_acumulado'].apply(lambda x: f"{float(x):.2f}")
        df_view_saldo['percentual_compensacao'] = df_view_saldo['percentual_compensacao'].apply(lambda x: f"{float(x):.2f}%")
        st.dataframe(df_view_saldo, use_container_width=True)

    with tab3:
        # Prepara View Final
        df_view = df_rateio_final.copy()
        if usar_dados_banco:
            mapeamento = {
                'media_consumo_6m': 'Média Consumo', 'credito_ideal_kwh': 'Crédito Ideal', 
                'meses_autonomia': 'Meses de Crédito', 'rateio_ideal_pct': 'Rateio Ideal %',
                'proximo_rateio_pct': 'Próximo Rateio %', 'observacao_gestao': 'Observações'
            }
            df_view = df_view.rename(columns=mapeamento)
        
        # --- FORMATAÇÃO VISUAL RIGOROSA (2 CASAS DECIMAIS) ---
        cols_float = ['Média Consumo', 'saldo_acumulado', 'Crédito Ideal', 'Meses de Crédito']
        for col in cols_float:
            if col in df_view.columns:
                df_view[col] = df_view[col].apply(lambda x: f"{float(x):.2f}")
        
        # Colunas Percentuais
        df_view['Rateio Ideal %'] = df_view['Rateio Ideal %'].apply(lambda x: f"{float(x):.2f}%")
        
        # Ajuste Rateio Atual
        if 'percentual_compensacao' in df_view.columns:
             df_view['Rateio Atual %'] = df_view['percentual_compensacao'].apply(lambda x: f"{float(x):.2f}%")
        
        df_view['Próximo Rateio %'] = df_view['Próximo Rateio %'].apply(lambda x: f"{float(x):.2f}%")

        # Seleção de Colunas
        colunas_finais = ['codigo_cliente', 'Média Consumo', 'saldo_acumulado', 'Crédito Ideal', 
                          'Meses de Crédito', 'Rateio Ideal %', 'Rateio Atual %', 'Próximo Rateio %', 'Observações']
        cols_validas = [c for c in colunas_finais if c in df_view.columns]
        df_view = df_view[cols_validas]
        
        df_view.columns = ['Cód. Cliente', 'Média de Consumo (6m)', 'Saldo (kWh)', 'Crédito Ideal', 
                           'Meses', 'Rateio Ideal %', 'Rateio Atual %', 'Próximo Rateio %', 'Observações']

        def style_rows(row):
            if "Alto" in str(row['Observações']): return ['background-color: #fff9c4'] * len(row)
            if "Baixo" in str(row['Observações']): return ['background-color: #ffcdd2'] * len(row)
            return [''] * len(row)

        st.dataframe(df_view.style.apply(style_rows, axis=1), use_container_width=True)

