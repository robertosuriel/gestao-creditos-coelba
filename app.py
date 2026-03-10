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
st.set_page_config(page_title="Casa do Ar - Gestão Energética", layout="wide")

# Configurações de conexão
DB_CONFIG = st.secrets["database"]["url"]
MINHA_API_KEY = st.secrets["gemini"]["api_key"]
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
    """Verifica se esta usina específica já possui este período processado."""
    query = text("""
        SELECT c.id_auditoria 
        FROM audit_fatura_cabecalho c
        INNER JOIN audit_fatura_analise_rateio r ON c.id_auditoria = r.id_auditoria
        WHERE c.geradora_nome = :gn AND c.periodo_apuracao = :per
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"gn": geradora_nome, "per": periodo}).fetchone()
        return int(result[0]) if result else None

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
        # Identifica a usina principal para calcular a média de geração
        id_usina_geradora = df_hist.groupby('codigo_cliente')['injetado'].sum().idxmax()
        media_ger = float(df_hist[df_hist['codigo_cliente'] == id_usina_geradora]['injetado'].tail(6).mean())
        
        # Inserção no Cabeçalho
        res = conn.execute(text("""
            INSERT INTO audit_fatura_cabecalho (geradora_nome, periodo_apuracao, media_geracao_mensal)
            VALUES (:gn, :per, :med) RETURNING id_auditoria
        """), {"gn": meta.get('geradora_nome'), "per": meta.get('periodo'), "med": media_ger})
        id_auditoria = res.fetchone()[0]

        # Inserção no Histórico
        df_h = df_hist.copy()
        df_h['id_auditoria'] = id_auditoria
        if 'data_ref' not in df_h.columns: 
            df_h['data_ref'] = df_h['situacao_mensal'].apply(parse_date)
        df_h = df_h.rename(columns={'consumo': 'consumo_kwh', 'injetado': 'injetado_kwh', 'data_ref': 'data_referencia'})
        df_h[['id_auditoria', 'situacao_mensal', 'codigo_cliente', 'consumo_kwh', 'injetado_kwh', 'data_referencia']].to_sql('audit_fatura_historico', conn, if_exists='append', index=False)

        # Inserção na Análise de Rateio
        df_r = df_rateio.copy()
        df_r['id_auditoria'] = id_auditoria
        df_r['rateio_atual_decimal'] = df_r['percentual_compensacao'] / 100
        df_r['rateio_ideal_decimal'] = df_r['Rateio Ideal %'] / 100
        df_r['proximo_rateio_decimal'] = df_r['Próximo Rateio %'] / 100
        
        map_r = {'saldo_acumulado': 'saldo_acumulado_kwh', 'Média Consumo': 'media_consumo_6m',
                 'Crédito Ideal': 'credito_ideal_kwh', 'Meses de Crédito': 'meses_autonomia', 'Observações': 'observacao_gestao'}
        df_r = df_r.rename(columns=map_r)
        
        cols_r = ['id_auditoria', 'codigo_cliente', 'saldo_acumulado_kwh', 'media_consumo_6m', 'credito_ideal_kwh', 
                'meses_autonomia', 'rateio_atual_decimal', 'rateio_ideal_decimal', 'proximo_rateio_decimal', 'observacao_gestao']
        df_r[[c for c in cols_r if c in df_r.columns]].to_sql('audit_fatura_analise_rateio', conn, if_exists='append', index=False)
    return id_auditoria

# =================================================================
# 3. INTEGRAÇÃO COM GEMINI (IA)
# =================================================================

@st.cache_data(show_spinner=False)
def process_with_gemini(file_bytes, api_key):
    client = genai.Client(api_key=api_key)
    prompt = """
    Extraia os dados do PDF estritamente no formato JSON.
    IMPORTANTE: No campo 'geradora_nome', capture o nome completo E O CÓDIGO NUMÉRICO (Ex: NCA ENERGIA 7093810311).
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
    Analise: {dados_json}
    Sugira o 'Próximo Rateio %' para equilibrar créditos da usina.
    META: 2 meses de autonomia. Unidade Geradora ({id_geradora}) deve ser 0.0.
    Retorne JSON: {{"codigo_cliente": valor_float}}
    """
    resp = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt_ia)
    try:
        sugestoes = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
        # Normalização Matemática para 100%
        sugestoes[id_geradora] = 0.0
        soma = sum(sugestoes.values())
        if soma > 0:
            for k in sugestoes: sugestoes[k] = round((sugestoes[k] / soma) * 100, 2)
        return sugestoes
    except: return {}

def clean_val(val):
    if val is None or val == "": return 0.0
    if isinstance(val, (int, float)): return float(val)
    return float(str(val).replace('.', '').replace(',', '.'))

def parse_date(date_str):
    try:
        parts = date_str.split('/')
        mes_en = MESES_MAP.get(parts[0].lower(), parts[0])
        return pd.to_datetime(f"{mes_en}/{parts[1]}", format='%b/%Y')
    except: return pd.NaT

# =================================================================
# 4. INTERFACE E DASHBOARD (STREAMLIT)
# =================================================================

st.title("📊 Blue Solutions - Gestão Energética")

df_relatorios = listar_relatorios_salvos()
opcao_selecionada = st.sidebar.selectbox("Carregar relatório:", options=["Novo Upload"] + df_relatorios['geradora_nome'].unique().tolist())

df_historico_raw, df_rateio_final, meta_dados = None, None, {}
usar_dados_banco = False

if opcao_selecionada != "Novo Upload":
    # Carregar do Banco
    row = df_relatorios[df_relatorios['geradora_nome'] == opcao_selecionada].iloc[0]
    id_auditoria = int(row['id_auditoria'])
    with st.spinner("Carregando banco AWS..."):
        df_historico_raw, df_rateio_final = carregar_dados_banco(id_auditoria)
        meta_dados = {'geradora_nome': row['geradora_nome'], 'periodo': row['periodo_apuracao']}
        usar_dados_banco = True
else:
    # Processar Novo Upload
    uploaded_files = st.file_uploader("Arraste PDFs Coelba", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        selected_file = st.sidebar.selectbox("Arquivo Atual:", [f.name for f in uploaded_files])
        file_obj = next(f for f in uploaded_files if f.name == selected_file)
        
        with st.spinner("IA Processando..."):
            dados_ia = process_with_gemini(file_obj.getvalue(), MINHA_API_KEY)
            meta_dados = dados_ia.get('metadata', {})
            
            # CHAVE DE DUPLICIDADE CORRIGIDA
            id_existente = verificar_relatorio_completo(meta_dados.get('geradora_nome'), meta_dados.get('periodo'))
            
            if id_existente:
                st.info("🔄 Relatório já existente. Carregando dados salvos...")
                df_historico_raw, df_rateio_final = carregar_dados_banco(id_existente)
                usar_dados_banco = True
            else:
                st.warning("⚡ Novo relatório detectado!")
                df_historico_raw = pd.DataFrame(dados_ia['tabela_13_meses'])
                df_saldo_raw = pd.DataFrame(dados_ia['resumo_saldo'])
                for col in ['consumo', 'injetado']: df_historico_raw[col] = df_historico_raw[col].apply(clean_val)
                for col in ['saldo_acumulado', 'percentual_compensacao']: df_saldo_raw[col] = df_saldo_raw[col].apply(clean_val)
                df_rateio_final = df_saldo_raw.copy()

# Renderização do Dashboard
if df_historico_raw is not None:
    df_historico_raw['data_ref'] = df_historico_raw['situacao_mensal'].apply(parse_date)
    df_historico_raw = df_historico_raw.sort_values('data_ref')
    id_geradora = df_historico_raw.groupby('codigo_cliente')['injetado'].sum().idxmax()
    
    # Cálculos se for Novo Upload
    if not usar_dados_banco:
        media_ger = df_historico_raw[df_historico_raw['codigo_cliente'] == id_geradora]['injetado'].tail(6).mean()
        avg_c = df_historico_raw[df_historico_raw['codigo_cliente'] != id_geradora].groupby('codigo_cliente')['consumo'].apply(lambda x: x.tail(6).mean()).reset_index()
        
        df_rateio_final = df_rateio_final.merge(avg_c, on='codigo_cliente', how='left').rename(columns={'consumo': 'Média Consumo'})
        df_rateio_final['Crédito Ideal'] = df_rateio_final['Média Consumo'] * 2
        df_rateio_final['Meses de Crédito'] = (df_rateio_final['saldo_acumulado'] / df_rateio_final['Média Consumo']).replace([float('inf')], 0).fillna(0)
        df_rateio_final['Rateio Ideal %'] = (df_rateio_final['Média Consumo'] / media_ger * 100)
        
        # IA Sugestão
        sugestoes = sugerir_proximo_rateio(df_rateio_final.to_json(), id_geradora, MINHA_API_KEY)
        df_rateio_final['Próximo Rateio %'] = df_rateio_final['codigo_cliente'].map(sugestoes).fillna(df_rateio_final['Rateio Ideal %'])
        
        def definir_obs(row):
            if row['Meses de Crédito'] > 2.5: return "Crédito Alto: Sugerido reduzir rateio"
            if row['Meses de Crédito'] < 1.7 and row['codigo_cliente'] != id_geradora: return "Crédito Baixo: Sugerido aumentar rateio"
            return "Equilibrado"
        df_rateio_final['Observações'] = df_rateio_final.apply(definir_obs, axis=1)
        
        salvar_processamento(meta_dados, df_historico_raw, df_rateio_final)

    # UI Gráficos
    st.info(f"Geradora: **{meta_dados.get('geradora_nome')}** | Período: **{meta_dados.get('periodo')}**")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.line(df_historico_raw[df_historico_raw['codigo_cliente'] == id_geradora], x='situacao_mensal', y='injetado', title="Histórico Geração", color_discrete_sequence=['green']), use_container_width=True)
    with c2: st.plotly_chart(px.line(df_historico_raw[df_historico_raw['codigo_cliente'] != id_geradora], x='situacao_mensal', y='consumo', color='codigo_cliente', title="Histórico Consumo"), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📋 Histórico", "💰 Saldos", "🎯 Análise de Rateio"])
    with tab1: st.dataframe(df_historico_raw, use_container_width=True)
    with tab2: st.dataframe(df_rateio_final[['codigo_cliente', 'saldo_acumulado', 'percentual_compensacao']], use_container_width=True)
    with tab3:
        df_view = df_rateio_final.copy()
        if usar_dados_banco:
            df_view = df_view.rename(columns={'media_consumo_6m': 'Média Consumo', 'rateio_ideal_pct': 'Rateio Ideal %', 'proximo_rateio_pct': 'Próximo Rateio %', 'observacao_gestao': 'Observações'})
        
        # Formatação e Cores
        for col in ['Média Consumo', 'saldo_acumulado', 'Crédito Ideal', 'Meses de Crédito']: 
            if col in df_view.columns: df_view[col] = df_view[col].map('{:,.2f}'.format)
        
        st.dataframe(df_view.style.apply(lambda r: ['background-color: #fff9c4' if 'Alto' in str(r.Observações) else 'background-color: #ffcdd2' if 'Baixo' in str(r.Observações) else '' for _ in r], axis=1), use_container_width=True)
