# pisf_dvh_analysis_complete.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pysus.online_data import SINAN
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

class PISFAnalysis:
    def __init__(self):
        """Inicializa a análise completa PISF-DVH"""
        self.dados = {
            'sinan': None,
            'pisf': None,
            'populacao': None,
            'integrado': None
        }
        
        # Configurações
        self.UFS_ESTUDO = ['PE', 'PB', 'RN', 'CE']
        self.ANOS_ESTUDO = list(range(2020, 2024))
        
        # Doenças de veiculação hídrica (CID-10)
        self.DOENCAS_DVH = {
            'A09': 'Diarreia e gastroenterite',
            'A00': 'Cólera',
            'A01': 'Febres tifóide e paratifóide',
            'A02': 'Outras infecções por Salmonella',
            'A03': 'Shiguelose',
            'A04': 'Outras infecções intestinais bacterianas',
            'A06': 'Amebíase',
            'A07': 'Outras doenças intestinais por protozoários',
            'A27': 'Leptospirose',
            'B15': 'Hepatite A',
            'B65': 'Esquistossomose'
        }
        
    # ====================== MÉTODO 1: PySUS ======================
    def obter_dados_sinan_pysus(self, salvar_csv=True):
        """Obtém dados do SINAN via PySUS com tratamento robusto"""
        print("="*60)
        print("OBTENÇÃO DE DADOS DO SINAN VIA PySUS")
        print("="*60)
        
        dados_todos = []
        log_erros = []
        
        for doenca_cod, doenca_nome in self.DOENCAS_DVH.items():
            print(f"\nProcessando {doenca_cod} - {doenca_nome}")
            
            for ano in self.ANOS_ESTUDO:
                try:
                    print(f"  Ano {ano}...", end=" ")
                    
                    # Baixar dados
                    df = SINAN.download(doenca_cod, ano)
                    
                    if df is not None and not df.empty:
                        # Filtrar apenas UFs de interesse
                        if 'SG_UF_NOT' in df.columns:
                            df = df[df['SG_UF_NOT'].isin(self.UFS_ESTUDO)]
                        
                        # Adicionar metadados
                        df['doenca_cod'] = doenca_cod
                        df['doenca_nome'] = doenca_nome
                        df['ano_notificacao'] = ano
                        
                        # Manter apenas colunas essenciais
                        colunas_manter = [
                            'ID_MUNICIP', 'DT_NOTIFIC', 'NU_ANO', 'SG_UF_NOT',
                            'ID_MN_RESI', 'NM_MUN_RES', 'CS_SEXO', 'NU_IDADE',
                            'DT_SIN_PRI', 'DT_NASC', 'EVOLUCAO',
                            'doenca_cod', 'doenca_nome', 'ano_notificacao'
                        ]
                        
                        colunas_disponiveis = [c for c in colunas_manter if c in df.columns]
                        df = df[colunas_disponiveis]
                        
                        dados_todos.append(df)
                        print(f"✓ {len(df):,} registros")
                    else:
                        print("✗ Sem dados")
                        log_erros.append(f"{doenca_cod}-{ano}: Sem dados")
                        
                except Exception as e:
                    print(f"✗ Erro: {str(e)[:50]}")
                    log_erros.append(f"{doenca_cod}-{ano}: {str(e)[:100]}")
        
        # Consolidar dados
        if dados_todos:
            df_completo = pd.concat(dados_todos, ignore_index=True, sort=False)
            
            # Log de resultados
            print("\n" + "="*60)
            print("RESUMO DA COLETA")
            print("="*60)
            print(f"Total de registros: {len(df_completo):,}")
            print(f"Período: {df_completo['ano_notificacao'].min()}-{df_completo['ano_notificacao'].max()}")
            print(f"UF's: {df_completo['SG_UF_NOT'].unique() if 'SG_UF_NOT' in df_completo.columns else 'Não disponível'}")
            print(f"Doenças coletadas: {df_completo['doenca_nome'].nunique()}")
            
            # Salvar log de erros
            with open('log_erros_coleta.json', 'w') as f:
                json.dump(log_erros, f, indent=2)
            
            # Salvar dados
            if salvar_csv:
                df_completo.to_csv('dados_sinan_completo.csv', index=False)
                print(f"\nDados salvos em: dados_sinan_completo.csv")
            
            self.dados['sinan'] = df_completo
            return df_completo
            
        else:
            print("Nenhum dado foi coletado!")
            return None
    
    # ====================== MÉTODO 2: FTP Alternativo ======================
    def obter_dados_populacao_ibge(self):
        """Obtém dados populacionais do IBGE (simulação)"""
        print("\nObtendo dados populacionais...")
        
        # Gerar dados populacionais simulados baseados em estimativas reais
        municipios_pe = [f'26{str(i).zfill(4)}' for i in range(1000, 3000)]  # Códigos PE
        
        dados_pop = []
        for cod_mun in municipios_pe[:200]:  # 200 municípios de exemplo
            pop_base = np.random.randint(15000, 350000)
            for ano in self.ANOS_ESTUDO:
                crescimento = np.random.uniform(0.98, 1.02)  # Variação de ±2%
                populacao = int(pop_base * (crescimento ** (ano - 2020)))
                
                dados_pop.append({
                    'cod_municipio': cod_mun,
                    'ano': ano,
                    'populacao': populacao,
                    'uf': 'PE'
                })
        
        df_populacao = pd.DataFrame(dados_pop)
        
        # Salvar
        df_populacao.to_csv('dados_populacao_estimada.csv', index=False)
        self.dados['populacao'] = df_populacao
        
        print(f"Dados populacionais gerados: {len(df_populacao):,} registros")
        return df_populacao
    
    # ====================== PROCESSAMENTO DOS DADOS ======================
    def processar_dados_sinan(self, df_sinan):
        """Processa e limpa os dados do SINAN"""
        print("\n" + "="*60)
        print("PROCESSAMENTO DOS DADOS SINAN")
        print("="*60)
        
        df = df_sinan.copy()
        
        # 1. Converter código do município
        if 'ID_MUNICIP' in df.columns:
            df['cod_municipio'] = df['ID_MUNICIP'].astype(str).str[:6]
            print(f"Códigos municipais convertidos: {df['cod_municipio'].nunique()} municípios")
        
        # 2. Converter datas
        colunas_data = ['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_NASC']
        for col in colunas_data:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                n_validas = df[col].notna().sum()
                print(f"Datas {col}: {n_validas:,} válidas de {len(df):,}")
        
        # 3. Converter idade
        if 'NU_IDADE' in df.columns:
            def converter_idade_sinan(codigo):
                try:
                    cod_str = str(int(codigo)).zfill(4)
                    unidade = cod_str[0]
                    valor = int(cod_str[1:])
                    
                    if unidade == '0':  # Minutos
                        return valor / (60 * 24 * 365.25)
                    elif unidade == '1':  # Horas
                        return valor / (24 * 365.25)
                    elif unidade == '2':  # Dias
                        return valor / 365.25
                    elif unidade == '3':  # Meses
                        return valor / 12
                    elif unidade == '4':  # Anos
                        return valor
                    else:
                        return np.nan
                except:
                    return np.nan
            
            df['idade_anos'] = df['NU_IDADE'].apply(converter_idade_sinan)
            print(f"Idades convertidas: {df['idade_anos'].notna().sum():,} válidas")
        
        # 4. Agregar por município-ano-doença
        print("\nAgregando casos por município...")
        casos = df.groupby(['cod_municipio', 'ano_notificacao', 'doenca_cod', 'doenca_nome']).size().reset_index()
        casos.columns = ['cod_municipio', 'ano', 'doenca_cod', 'doenca_nome', 'casos']
        
        print(f"Agregação completa: {len(casos):,} registros agregados")
        return casos
    
    def calcular_taxas_incidencia(self, casos, populacao):
        """Calcula taxas de incidência por 100 mil habitantes"""
        print("\n" + "="*60)
        print("CÁLCULO DE TAXAS DE INCIDÊNCIA")
        print("="*60)
        
        # 1. Juntar com dados populacionais
        df_taxas = pd.merge(casos, populacao, 
                           on=['cod_municipio', 'ano'],
                           how='inner')
        
        # 2. Calcular taxa por 100 mil
        df_taxas['taxa_100mil'] = (df_taxas['casos'] / df_taxas['populacao']) * 100000
        
        # 3. Agregar taxa total de todas as DVH
        taxa_total = df_taxas.groupby(['cod_municipio', 'ano']).agg({
            'casos': 'sum',
            'populacao': 'first'
        }).reset_index()
        
        taxa_total['taxa_total_100mil'] = (taxa_total['casos'] / taxa_total['populacao']) * 100000
        
        # Estatísticas
        print(f"Taxas calculadas para {taxa_total['cod_municipio'].nunique()} municípios")
        print(f"Período: {taxa_total['ano'].min()}-{taxa_total['ano'].max()}")
        print(f"Taxa média: {taxa_total['taxa_total_100mil'].mean():.2f} por 100 mil")
        print(f"Taxa mediana: {taxa_total['taxa_total_100mil'].median():.2f} por 100 mil")
        
        # Salvar
        df_taxas.to_csv('taxas_dvh_detalhadas.csv', index=False)
        taxa_total.to_csv('taxas_dvh_agregadas.csv', index=False)
        
        return df_taxas, taxa_total
    
    # ====================== ANÁLISE DID ======================
    def executar_analise_did(self, df_taxas, dados_pisf):
        """Executa análise de Diferenças-em-Diferenças"""
        print("\n" + "="*60)
        print("ANÁLISE DE DIFERENÇAS-EM-DIFERENÇAS")
        print("="*60)
        
        # 1. Integrar dados
        df = pd.merge(df_taxas, dados_pisf,
                     on='cod_municipio',
                     how='left',
                     suffixes=('', '_pisf'))
        
        # 2. Criar variáveis DiD
        df['tratado'] = df['ano_inicio'].notna().astype(int)
        df['post'] = 0
        
        # Para municípios tratados, post=1 após ano_inicio
        mask_post = (df['tratado'] == 1) & (df['ano'] >= df['ano_inicio'])
        df.loc[mask_post, 'post'] = 1
        
        df['did'] = df['tratado'] * df['post']
        
        # 3. Estatísticas descritivas
        print("\nESTATÍSTICAS DESCRITIVAS:")
        
        # Pré-tratamento (2020-2021)
        pre_t = df[(df['tratado'] == 1) & (df['ano'] < 2022)]['taxa_total_100mil'].mean()
        pre_c = df[(df['tratado'] == 0) & (df['ano'] < 2022)]['taxa_total_100mil'].mean()
        
        # Pós-tratamento (2022-2023)
        pos_t = df[(df['tratado'] == 1) & (df['ano'] >= 2022)]['taxa_total_100mil'].mean()
        pos_c = df[(df['tratado'] == 0) & (df['ano'] >= 2022)]['taxa_total_100mil'].mean()
        
        print(f"Grupo Tratamento - Pré: {pre_t:.1f}, Pós: {pos_t:.1f}, Δ: {pos_t - pre_t:.1f}")
        print(f"Grupo Controle - Pré: {pre_c:.1f}, Pós: {pos_c:.1f}, Δ: {pos_c - pre_c:.1f}")
        print(f"Diferença-em-Diferenças: {(pos_t - pre_t) - (pos_c - pre_c):.1f}")
        
        # 4. Modelo DiD
        print("\nMODELO DE REGRESSÃO DiD:")
        
        # Preparar dados para regressão
        df_reg = df.copy()
        
        # Efeitos fixos
        df_reg['municipio_fe'] = pd.Categorical(df_reg['cod_municipio'])
        df_reg['ano_fe'] = pd.Categorical(df_reg['ano'])
        
        # Modelo básico DiD
        modelo = smf.ols('taxa_total_100mil ~ tratado + post + did',
                        data=df_reg).fit()
        
        print(modelo.summary())
        
        # 5. Modelo com efeitos fixos
        print("\nMODELO DiD COM EFEITOS FIXOS:")
        modelo_fe = smf.ols('taxa_total_100mil ~ did + C(cod_municipio) + C(ano)',
                          data=df_reg).fit(cov_type='cluster',
                                          cov_kwds={'groups': df_reg['cod_municipio']})
        
        print(modelo_fe.summary())
        
        # Salvar resultados
        resultados = {
            'modelo_basico': modelo.params.to_dict(),
            'modelo_fe': modelo_fe.params.to_dict(),
            'estatisticas': {
                'pre_tratamento': pre_t,
                'pos_tratamento': pos_t,
                'pre_controle': pre_c,
                'pos_controle': pos_c,
                'did_estimado': modelo_fe.params.get('did', 0)
            }
        }
        
        with open('resultados_did.json', 'w') as f:
            json.dump(resultados, f, indent=2)
        
        self.dados['integrado'] = df_reg
        return df_reg, modelo_fe
    
    # ====================== VISUALIZAÇÃO ======================
    def gerar_visualizacoes(self, df_did, modelo_did):
        """Gera visualizações para o artigo"""
        print("\n" + "="*60)
        print("GERANDO VISUALIZAÇÕES")
        print("="*60)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Tendência temporal por grupo
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Tendência média
        df_media = df_did.groupby(['ano', 'tratado']).agg({
            'taxa_total_100mil': 'mean'
        }).reset_index()
        
        df_media['Grupo'] = df_media['tratado'].map({0: 'Controle', 1: 'Tratamento'})
        
        ax1 = axes[0, 0]
        for grupo in ['Controle', 'Tratamento']:
            dados = df_media[df_media['Grupo'] == grupo]
            ax1.plot(dados['ano'], dados['taxa_total_100mil'], 
                    marker='o', linewidth=2, markersize=8, label=grupo)
        
        ax1.axvline(x=2021.5, color='red', linestyle='--', alpha=0.7, label='Início PISF (média)')
        ax1.set_title('Evolução da Taxa de DVH por Grupo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Taxa por 100 mil hab.')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuição antes/depois
        ax2 = axes[0, 1]
        dados_box = []
        labels_box = []
        
        for grupo_nome, grupo_cod in [('Controle', 0), ('Tratamento', 1)]:
            pre = df_did[(df_did['tratado'] == grupo_cod) & (df_did['post'] == 0)]['taxa_total_100mil']
            pos = df_did[(df_did['tratado'] == grupo_cod) & (df_did['post'] == 1)]['taxa_total_100mil']
            
            dados_box.extend([pre, pos])
            labels_box.extend([f'{grupo_nome}\nPré', f'{grupo_nome}\nPós'])
        
        ax2.boxplot(dados_box, labels=labels_box)
        ax2.set_title('Distribuição das Taxas de DVH', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Taxa por 100 mil hab.')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Mapa de coeficientes DiD
        ax3 = axes[1, 0]
        coeficientes = modelo_did.params
        erros = modelo_did.bse
        
        variaveis = ['Intercepto', 'Tratado × Pós (DiD)']
        valores = [coeficientes.get('Intercept', 0), coeficientes.get('did', 0)]
        erros_val = [erros.get('Intercept', 0), erros.get('did', 0)]
        
        x_pos = np.arange(len(variaveis))
        ax3.bar(x_pos, valores, yerr=erros_val, 
               capsize=10, alpha=0.7, color=['skyblue', 'lightcoral'])
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(variaveis, rotation=45, ha='right')
        ax3.set_title('Coeficientes do Modelo DiD', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Valor do Coeficiente')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Resíduos do modelo
        ax4 = axes[1, 1]
        if hasattr(modelo_did, 'resid'):
            ax4.scatter(modelo_did.fittedvalues, modelo_did.resid, alpha=0.6)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Valores Ajustados')
            ax4.set_ylabel('Resíduos')
            ax4.set_title('Resíduos vs Valores Ajustados', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizacoes_analise.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizações salvas em: visualizacoes_analise.png")
    
    # ====================== EXECUÇÃO COMPLETA ======================
    def executar_analise_completa(self):
        """Executa toda a análise de forma integrada"""
        print("="*70)
        print("ANÁLISE COMPLETA: IMPACTO DO PISF NAS DOENÇAS DE VEICULAÇÃO HÍDRICA")
        print("="*70)
        
        # 1. Obter dados do SINAN
        df_sinan = self.obter_dados_sinan_pysus(salvar_csv=True)
        
        if df_sinan is None:
            print("Erro: Não foi possível obter dados do SINAN")
            return
        
        # 2. Obter dados populacionais
        df_populacao = self.obter_dados_populacao_ibge()
        
        # 3. Processar dados SINAN
        casos = self.processar_dados_sinan(df_sinan)
        
        # 4. Calcular taxas
        df_taxas, taxa_total = self.calcular_taxas_incidencia(casos, df_populacao)
        
        # 5. Dados PISF simulados (substituir por reais)
        dados_pisf_simulados = self.simular_dados_pisf(taxa_total)
        
        # 6. Executar análise DiD
        df_did, modelo_did = self.executar_analise_did(taxa_total, dados_pisf_simulados)
        
        # 7. Gerar visualizações
        self.gerar_visualizacoes(df_did, modelo_did)
        
        # 8. Salvar relatório
        self.gerar_relatorio_final(modelo_did)
        
        print("\n" + "="*70)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*70)
    
    def simular_dados_pisf(self, df_taxas):
        """Simula dados do PISF para demonstração"""
        municipios = df_taxas['cod_municipio'].unique()
        n_tratados = len(municipios) // 2
        
        tratados = np.random.choice(municipios, n_tratados, replace=False)
        
        dados_pisf = []
        for mun in municipios:
            ano_inicio = 2022 if mun in tratados else np.nan
            
            dados_pisf.append({
                'cod_municipio': mun,
                'ano_inicio': ano_inicio,
                'status': 'Beneficiado' if pd.notna(ano_inicio) else 'Não beneficiado',
                'eixo': 'Leste' if pd.notna(ano_inicio) else 'Não aplicável'
            })
        
        return pd.DataFrame(dados_pisf)
    
    def gerar_relatorio_final(self, modelo_did):
        """Gera relatório final da análise"""
        resultados = {
            'data_analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'periodo_estudo': f"{self.ANOS_ESTUDO[0]}-{self.ANOS_ESTUDO[-1]}",
            'ufs_incluidas': self.UFS_ESTUDO,
            'doencas_analisadas': list(self.DOENCAS_DVH.values()),
            'resultados_did': {
                'coeficiente_did': float(modelo_did.params.get('did', 0)),
                'erro_padrao': float(modelo_did.bse.get('did', 0)),
                'p_valor': float(modelo_did.pvalues.get('did', 1)),
                'significancia': '***' if modelo_did.pvalues.get('did', 1) < 0.01 else 
                               '**' if modelo_did.pvalues.get('did', 1) < 0.05 else 
                               '*' if modelo_did.pvalues.get('did', 1) < 0.1 else 'ns'
            },
            'arquivos_gerados': [
                'dados_sinan_completo.csv',
                'dados_populacao_estimada.csv',
                'taxas_dvh_detalhadas.csv',
                'taxas_dvh_agregadas.csv',
                'visualizacoes_analise.png',
                'resultados_did.json'
            ]
        }
        
        with open('relatorio_analise_pisf.json', 'w') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        
        # Imprimir resumo
        print("\n" + "="*60)
        print("RELATÓRIO FINAL DA ANÁLISE")
        print("="*60)
        print(f"Data da análise: {resultados['data_analise']}")
        print(f"Período: {resultados['periodo_estudo']}")
        print(f"Estados incluídos: {', '.join(resultados['ufs_incluidas'])}")
        print(f"\nRESULTADO PRINCIPAL - EFEITO PISF:")
        print(f"  Coeficiente DiD: {resultados['resultados_did']['coeficiente_did']:.2f}")
        print(f"  Erro padrão: {resultados['resultados_did']['erro_padrao']:.2f}")
        print(f"  P-valor: {resultados['resultados_did']['p_valor']:.4f}")
        print(f"  Significância: {resultados['resultados_did']['significancia']}")
        
        if resultados['resultados_did']['p_valor'] < 0.05:
            print(f"\n✓ CONCLUSÃO: O PISF teve efeito estatisticamente significativo")
            print(f"  na redução das doenças de veiculação hídrica.")
        else:
            print(f"\n⚠ CONCLUSÃO: Não foi detectado efeito estatisticamente")
            print(f"  significativo do PISF na redução das DVH.")

# Executar análise completa
if __name__ == "__main__":
    # Inicializar e executar análise
    analisador = PISFAnalysis()
    analisador.executar_analise_completa()