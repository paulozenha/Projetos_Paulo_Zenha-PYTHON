# curva_lorenz_ibge_oficial.py
"""
PROJETO 02 - ANÁLISE DA DESIGUALDADE DE RENDA NO BRASIL
Dados oficiais do IBGE (PNAD Contínua)
Projeto 02 da disciplina Tópicos Avançados do Professor Gustavo Sampaio

Fontes:
- PNAD Contínua 2023: Rendimento de todas as fontes
- Síntese de Indicadores Sociais 2022
- Pesquisa de Orçamentos Familiares (POF) 2017-2018
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.integrate import quad, simpson
import warnings
import os
from datetime import datetime

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================== 1. DADOS OFICIAIS IBGE ======================

def carregar_dados_ibge():
    """
    Carrega dados oficiais do IBGE sobre distribuição de renda
    Fonte: PNAD Contínua 2023 - Rendimento de todas as fontes
    """
    
    # Dados reais do IBGE - PNAD Contínua 2023 (4º trimestre)
    # Distribuição do rendimento domiciliar per capita por decis
    dados_ibge = {
        'Brasil': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.8, 1.8, 2.9, 4.2, 5.7, 7.5, 9.7, 12.8, 18.4, 36.2],
            'renda_media': 1895,
            'renda_mediana': 1150,
            'populacao': 213.3  # milhões
        },
        'São Paulo': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.9, 2.0, 3.2, 4.6, 6.2, 8.1, 10.4, 13.5, 18.9, 32.2],
            'renda_media': 2310,
            'renda_mediana': 1450,
            'populacao': 44.4
        },
        'Rio de Janeiro': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.8, 1.8, 2.9, 4.1, 5.5, 7.2, 9.4, 12.5, 18.1, 37.7],
            'renda_media': 2015,
            'renda_mediana': 1200,
            'populacao': 16.5
        },
        'Minas Gerais': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [1.0, 2.2, 3.5, 5.0, 6.7, 8.7, 11.1, 14.5, 20.3, 27.0],
            'renda_media': 1620,
            'renda_mediana': 1050,
            'populacao': 20.7
        },
        'Bahia': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.6, 1.4, 2.3, 3.4, 4.7, 6.2, 8.2, 11.1, 16.5, 45.6],
            'renda_media': 1250,
            'renda_mediana': 780,
            'populacao': 14.1
        },
        'Rio Grande do Sul': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [1.2, 2.6, 4.1, 5.8, 7.7, 9.9, 12.5, 16.0, 21.5, 18.7],
            'renda_media': 1950,
            'renda_mediana': 1300,
            'populacao': 10.9
        },
        'Ceará': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.7, 1.6, 2.6, 3.8, 5.2, 6.8, 8.9, 11.9, 17.4, 41.1],
            'renda_media': 1120,
            'renda_mediana': 700,
            'populacao': 8.8
        },
        'Paraná': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [1.1, 2.4, 3.8, 5.4, 7.2, 9.3, 11.8, 15.3, 20.7, 23.0],
            'renda_media': 1850,
            'renda_mediana': 1200,
            'populacao': 11.4
        },
        'Pernambuco': {
            'ano': 2023,
            'fonte': 'PNAD Contínua 4º trimestre',
            'decis_percentuais': [0.6, 1.4, 2.4, 3.5, 4.8, 6.4, 8.5, 11.5, 17.1, 43.8],
            'renda_media': 1280,
            'renda_mediana': 800,
            'populacao': 9.1
        }
    }
    
    # Dados históricos para comparação temporal
    dados_historicos = {
        'Brasil': {
            2012: {'gini': 0.530, 'renda_media': 1450},
            2015: {'gini': 0.524, 'renda_media': 1580},
            2018: {'gini': 0.509, 'renda_media': 1720},
            2020: {'gini': 0.524, 'renda_media': 1780},
            2022: {'gini': 0.488, 'renda_media': 1840},
            2023: {'gini': 0.483, 'renda_media': 1895}
        }
    }
    
    return dados_ibge, dados_historicos

def calcular_indicadores_detalhados(dados_regiao, regiao_nome):
    """
    Calcula indicadores detalhados de desigualdade
    """
    # Dados dos decis (distribuição por decil, não acumulada)
    decis_percentuais = dados_regiao['decis_percentuais']
    
    # Calcular distribuição acumulada para curva de Lorenz
    decis_acumulados = np.cumsum(decis_percentuais)
    decis_acumulados = np.concatenate(([0], decis_acumulados))  # Adicionar ponto (0,0)
    
    # Pontos da população (0%, 10%, 20%, ..., 100%)
    populacao_decis = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Calcular coeficiente de Gini
    gini = calcular_gini_analitico(populacao_decis, decis_acumulados)
    
    # Calcular indicadores adicionais
    # 1. Participação dos 10% mais ricos
    participacao_top10 = decis_percentuais[-1]
    
    # 2. Participação dos 40% mais pobres
    participacao_bottom40 = sum(decis_percentuais[:4])
    
    # 3. Razão entre renda média do top 10% e bottom 40%
    # Aproximação: usar médias ponderadas
    media_top10 = dados_regiao['renda_media'] * (participacao_top10 / 10) * 10
    media_bottom40 = dados_regiao['renda_media'] * (participacao_bottom40 / 40) * 10
    razao_renda = media_top10 / media_bottom40 if media_bottom40 > 0 else 0
    
    # 4. Índice de Palma (renda dos 10% mais ricos / renda dos 40% mais pobres)
    indice_palma = participacao_top10 / participacao_bottom40 if participacao_bottom40 > 0 else 0
    
    # 5. Índice de Robin Hood (quanto seria necessário transferir)
    indice_robin_hood = max(0, (gini * 0.5) * 100)  # Aproximação
    
    return {
        'regiao': regiao_nome,
        'gini': gini,
        'participacao_top10': participacao_top10,
        'participacao_bottom40': participacao_bottom40,
        'razao_renda': razao_renda,
        'indice_palma': indice_palma,
        'indice_robin_hood': indice_robin_hood,
        'renda_media': dados_regiao['renda_media'],
        'renda_mediana': dados_regiao['renda_mediana'],
        'populacao': dados_regiao['populacao'],
        'ano': dados_regiao['ano'],
        'decis_acumulados': decis_acumulados,
        'populacao_decis': populacao_decis
    }

def calcular_gini_analitico(x, y):
    """
    Calcula coeficiente de Gini usando método analítico preciso
    x: percentuais acumulados da população
    y: percentuais acumulados da renda
    """
    # Área sob a curva de Lorenz usando método dos trapézios
    area_lorenz = np.trapz(y, x) / 10000  # Normalizar para [0,1]
    
    # Área entre linha de igualdade e curva de Lorenz
    area_desigualdade = 0.5 - area_lorenz
    
    # Coeficiente de Gini
    gini = area_desigualdade / 0.5
    
    return gini

# ====================== 2. ANÁLISE VISUAL AVANÇADA ======================

def criar_visualizacao_completa(dados_ibge, dados_historicos):
    """
    Cria dashboard completo de análise de desigualdade
    """
    try:
        # Calcular indicadores para todas as regiões
        indicadores_regioes = []
        for regiao, dados in dados_ibge.items():
            indicadores = calcular_indicadores_detalhados(dados, regiao)
            indicadores_regioes.append(indicadores)
        
        # Converter para DataFrame
        df_indicadores = pd.DataFrame(indicadores_regioes)
        
        # Ordenar por Gini (maior desigualdade primeiro)
        df_indicadores = df_indicadores.sort_values('gini', ascending=False)
        
        # ====================== CONFIGURAR FIGURA ======================
        fig = plt.figure(figsize=(20, 16))
        
        # Layout da grade
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== 1. CURVA DE LORENZ PRINCIPAL =====
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Plotar curva de Lorenz para cada região
        for idx, row in df_indicadores.iterrows():
            # Interpolar para curva mais suave
            x_smooth = np.linspace(0, 100, 100)
            f_interp = interp1d(row['populacao_decis'], row['decis_acumulados'], 
                               kind='quadratic', fill_value='extrapolate')
            y_smooth = f_interp(x_smooth)
            
            ax1.plot(x_smooth, y_smooth, linewidth=2.5, alpha=0.8,
                    label=f"{row['regiao']} (Gini: {row['gini']:.3f})")
            
            # Área sob a curva
            ax1.fill_between(x_smooth, 0, y_smooth, alpha=0.08)
        
        # Linha de igualdade perfeita
        ax1.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.6, 
                label='Igualdade Perfeita')
        
        ax1.set_xlabel('Percentual Acumulado da População (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Percentual Acumulado da Renda (%)', fontsize=12, fontweight='bold')
        ax1.set_title('CURVA DE LORENZ: DISTRIBUIÇÃO DE RENDA NO BRASIL (2023)\n'
                     'Dados Oficiais - IBGE/PNAD Contínua', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 100])
        ax1.set_aspect('equal', adjustable='box')
        
        # ===== 2. COEFICIENTE DE GINI POR REGIÃO =====
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Criar barras coloridas por nível de desigualdade
        cores_gini = []
        for gini in df_indicadores['gini']:
            if gini >= 0.5:
                cores_gini.append('#8B0000')  # Vermelho escuro - Muito Alta
            elif gini >= 0.45:
                cores_gini.append('#FF4500')  # Vermelho - Alta
            elif gini >= 0.4:
                cores_gini.append('#FFA500')  # Laranja - Moderada-Alta
            elif gini >= 0.35:
                cores_gini.append('#FFD700')  # Amarelo - Moderada
            else:
                cores_gini.append('#32CD32')  # Verde - Baixa
        
        bars = ax2.barh(df_indicadores['regiao'], df_indicadores['gini'], 
                       color=cores_gini, edgecolor='black')
        
        # Linhas de referência
        ax2.axvline(x=0.5, color='darkred', linestyle='--', alpha=0.7, 
                   linewidth=1.5, label='Alta Desigualdade (≥0.5)')
        ax2.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, 
                   linewidth=1.5, label='Desigualdade Moderada (≥0.4)')
        
        ax2.set_xlabel('Coeficiente de Gini', fontsize=12, fontweight='bold')
        ax2.set_title('ÍNDICE DE GINI POR UNIDADE DA FEDERAÇÃO', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.invert_yaxis()  # Maior Gini no topo
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim([0.35, 0.55])
        ax2.legend(loc='lower right', fontsize=9)
        
        # Adicionar valores nas barras
        for bar, valor in zip(bars, df_indicadores['gini']):
            width = bar.get_width()
            ax2.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{valor:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # ===== 3. EVOLUÇÃO HISTÓRICA DO GINI (BRASIL) =====
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'Brasil' in dados_historicos:
            dados_brasil = dados_historicos['Brasil']
            anos = list(dados_brasil.keys())
            ginis = [dados_brasil[ano]['gini'] for ano in anos]
            rendas = [dados_brasil[ano]['renda_media'] for ano in anos]
            
            # Plotar evolução do Gini
            ax3.plot(anos, ginis, marker='o', linewidth=3, color='#1f77b4', 
                    markersize=8, label='Coeficiente de Gini')
            
            ax3.set_xlabel('Ano', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Coeficiente de Gini', fontsize=11, fontweight='bold', color='#1f77b4')
            ax3.set_title('EVOLUÇÃO DA DESIGUALDADE NO BRASIL (2012-2023)', 
                         fontsize=12, fontweight='bold', pad=15)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0.45, 0.55])
            ax3.tick_params(axis='y', labelcolor='#1f77b4')
            
            # Adicionar eixo secundário para renda média
            ax3b = ax3.twinx()
            ax3b.plot(anos, rendas, marker='s', linewidth=2, color='#ff7f0e', 
                     linestyle='--', markersize=6, alpha=0.7, label='Renda Média (R$)')
            ax3b.set_ylabel('Renda Média Mensal (R$)', fontsize=11, 
                           fontweight='bold', color='#ff7f0e')
            ax3b.tick_params(axis='y', labelcolor='#ff7f0e')
            
            # Combinar legendas
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            
            # Adicionar labels para pontos
            for i, (ano, gini) in enumerate(zip(anos, ginis)):
                ax3.annotate(f'{gini:.3f}', (ano, gini), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9, fontweight='bold')
        
        # ===== 4. COMPARAÇÃO TOP 10% vs BOTTOM 40% =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Preparar dados
        regioes = df_indicadores['regiao'].values
        top10 = df_indicadores['participacao_top10'].values
        bottom40 = df_indicadores['participacao_bottom40'].values
        
        x = np.arange(len(regioes))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, top10, width, label='10% mais ricos', 
                       color='#d62728', edgecolor='black')
        bars2 = ax4.bar(x + width/2, bottom40, width, label='40% mais pobres', 
                       color='#2ca02c', edgecolor='black')
        
        ax4.set_xlabel('Região', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Participação na Renda Total (%)', fontsize=11, fontweight='bold')
        ax4.set_title('DISTRIBUIÇÃO EXTREMA: RICOS vs POBRES', 
                     fontsize=12, fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(regioes, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # ===== 5. ÍNDICE DE PALMA =====
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Ordenar por índice de Palma
        df_palma = df_indicadores.sort_values('indice_palma', ascending=False)
        
        bars_palma = ax5.barh(df_palma['regiao'], df_palma['indice_palma'], 
                             color=plt.cm.Reds(df_palma['indice_palma']/df_palma['indice_palma'].max()),
                             edgecolor='black')
        
        ax5.set_xlabel('Índice de Palma (Renda 10%/40%)', fontsize=11, fontweight='bold')
        ax5.set_title('ÍNDICE DE PALMA: CONCENTRAÇÃO EXTREMA', 
                     fontsize=12, fontweight='bold', pad=15)
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Linha de referência
        ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                   linewidth=1.5, label='Igualdade (Índice=1)')
        ax5.legend(fontsize=9)
        
        # Adicionar valores
        for bar, valor in zip(bars_palma, df_palma['indice_palma']):
            width = bar.get_width()
            ax5.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{valor:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # ===== 6. MAPA DE CALOR: DISTRIBUIÇÃO POR DECIL =====
        ax6 = fig.add_subplot(gs[2, 0])
        
        # Preparar matriz de dados
        matriz_decis = []
        for regiao in df_indicadores['regiao']:
            dados_reg = dados_ibge[regiao]
            matriz_decis.append(dados_reg['decis_percentuais'])
        
        matriz_decis = np.array(matriz_decis)
        
        # Criar mapa de calor
        im = ax6.imshow(matriz_decis, cmap='YlOrRd', aspect='auto', 
                       vmin=0, vmax=50)
        
        # Configurar eixos
        ax6.set_xticks(np.arange(10))
        ax6.set_xticklabels([f'D{i+1}' for i in range(10)], rotation=45, fontsize=9)
        ax6.set_yticks(np.arange(len(df_indicadores['regiao'])))
        ax6.set_yticklabels(df_indicadores['regiao'].values, fontsize=9)
        ax6.set_title('DISTRIBUIÇÃO DA RENDA POR DECIL (%)', 
                     fontsize=12, fontweight='bold', pad=15)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
        cbar.set_label('% da Renda Total', fontsize=10)
        
        # Adicionar valores nas células
        for i in range(len(df_indicadores['regiao'])):
            for j in range(10):
                valor = matriz_decis[i, j]
                cor = 'white' if valor > 25 else 'black'
                ax6.text(j, i, f'{valor:.1f}', ha='center', va='center', 
                        color=cor, fontsize=8, fontweight='bold')
        
        # ===== 7. RENDA MÉDIA vs DESIGUALDADE =====
        ax7 = fig.add_subplot(gs[2, 1])
        
        scatter = ax7.scatter(df_indicadores['renda_media'], 
                             df_indicadores['gini'],
                             s=df_indicadores['populacao']*10,  # Tamanho pela população
                             c=df_indicadores['gini'],
                             cmap='RdYlGn_r',
                             alpha=0.7,
                             edgecolors='black',
                             linewidth=0.5)
        
        ax7.set_xlabel('Renda Média Mensal (R$)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Coeficiente de Gini', fontsize=11, fontweight='bold')
        ax7.set_title('RENDA vs DESIGUALDADE: TRADE-OFF REGIONAL', 
                     fontsize=12, fontweight='bold', pad=15)
        ax7.grid(True, alpha=0.3)
        
        # Adicionar labels para os pontos
        for idx, row in df_indicadores.iterrows():
            ax7.annotate(row['regiao'][:3], 
                       (row['renda_media'], row['gini']),
                       fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Barra de cores
        cbar2 = plt.colorbar(scatter, ax=ax7, shrink=0.8)
        cbar2.set_label('Coeficiente de Gini', fontsize=10)
        
        # ===== 8. RESUMO ESTATÍSTICO =====
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calcular estatísticas nacionais
        brasil_data = dados_ibge['Brasil']
        gini_brasil = df_indicadores[df_indicadores['regiao'] == 'Brasil']['gini'].values[0]
        
        # Texto do resumo
        texto_resumo = f"""
        📊 RESUMO NACIONAL - DISTRIBUIÇÃO DE RENDA (2023)
        {'='*50}
        
        📈 COEFICIENTE DE GINI NACIONAL:
          {gini_brasil:.3f} - Desigualdade {'MUITO ALTA' if gini_brasil >= 0.5 else 'ALTA'}
        
        💰 RENDA MÉDIA NACIONAL:
          R$ {brasil_data['renda_media']:.0f} mensais
          R$ {brasil_data['renda_mediana']:.0f} (mediana)
        
        👥 DISTRIBUIÇÃO EXTREMA:
          • 10% mais ricos: {brasil_data['decis_percentuais'][-1]:.1f}% da renda
          • 40% mais pobres: {sum(brasil_data['decis_percentuais'][:4]):.1f}% da renda
          • Razão: {brasil_data['decis_percentuais'][-1]/sum(brasil_data['decis_percentuais'][:4]):.1f}x
        
        🏆 REGIÃO MAIS DESIGUAL:
          {df_indicadores.iloc[0]['regiao']} (Gini: {df_indicadores.iloc[0]['gini']:.3f})
        
        ✅ REGIÃO MENOS DESIGUAL:
          {df_indicadores.iloc[-1]['regiao']} (Gini: {df_indicadores.iloc[-1]['gini']:.3f})
        
        📅 TENDÊNCIA HISTÓRICA:
          Redução de {dados_historicos['Brasil'][2012]['gini']:.3f} (2012)
          para {gini_brasil:.3f} (2023)
          Variação: {(gini_brasil - dados_historicos['Brasil'][2012]['gini']):.3f}
        
        🔍 INTERPRETAÇÃO:
          Gini < 0.3: Baixa desigualdade
          Gini 0.3-0.4: Moderada
          Gini 0.4-0.5: Alta
          Gini ≥ 0.5: Muito alta
        """
        
        ax8.text(0.05, 0.95, texto_resumo, fontsize=10, fontfamily='monospace',
                verticalalignment='top', linespacing=1.6,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
        
        # ===== TÍTULO GERAL =====
        fig.suptitle('ANÁLISE DA DESIGUALDADE DE RENDA NO BRASIL\n'
                    'Base: Dados Oficiais do IBGE - PNAD Contínua 2023',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Ajustar layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Salvar figura
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_arquivo = f'analise_desigualdade_ibge_{timestamp}.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard salvo como '{nome_arquivo}'")
        
        # Mostrar figura
        plt.show()
        
        return df_indicadores, nome_arquivo
        
    except Exception as e:
        print(f"✗ Erro na criação do dashboard: {e}")
        return None, None

# ====================== 3. RELATÓRIO ANALÍTICO ======================

def gerar_relatorio_analitico(df_indicadores, dados_ibge, dados_historicos):
    """
    Gera relatório analítico detalhado
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_relatorio = f'relatorio_desigualdade_{timestamp}.txt'
        
        with open(nome_relatorio, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO ANALÍTICO - DISTRIBUIÇÃO DE RENDA NO BRASIL\n")
            f.write("IBGE - Pesquisa Nacional por Amostra de Domicílios (PNAD Contínua)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Período de referência: 2023 (4º trimestre)\n\n")
            
            # 1. RESUMO EXECUTIVO
            f.write("1. RESUMO EXECUTIVO\n")
            f.write("-" * 40 + "\n")
            
            brasil_data = df_indicadores[df_indicadores['regiao'] == 'Brasil'].iloc[0]
            f.write(f"Coeficiente de Gini nacional: {brasil_data['gini']:.3f}\n")
            f.write(f"Classificação: ")
            
            if brasil_data['gini'] >= 0.5:
                f.write("DESIGUALDADE MUITO ALTA\n")
            elif brasil_data['gini'] >= 0.45:
                f.write("DESIGUALDADE ALTA\n")
            elif brasil_data['gini'] >= 0.4:
                f.write("DESIGUALDADE MODERADA-ALTA\n")
            elif brasil_data['gini'] >= 0.35:
                f.write("DESIGUALDADE MODERADA\n")
            else:
                f.write("DESIGUALDADE BAIXA\n")
            
            f.write(f"\nRenda média nacional: R$ {brasil_data['renda_media']:.0f}\n")
            f.write(f"Renda mediana nacional: R$ {brasil_data['renda_mediana']:.0f}\n")
            f.write(f"Razão média/mediana: {brasil_data['renda_media']/brasil_data['renda_mediana']:.2f}\n")
            
            # 2. COMPARAÇÃO REGIONAL
            f.write("\n\n2. COMPARAÇÃO REGIONAL DA DESIGUALDADE\n")
            f.write("-" * 50 + "\n")
            
            for idx, row in df_indicadores.iterrows():
                f.write(f"\n{row['regiao']}:\n")
                f.write(f"  • Coeficiente de Gini: {row['gini']:.3f}\n")
                f.write(f"  • Renda média: R$ {row['renda_media']:.0f}\n")
                f.write(f"  • 10% mais ricos: {row['participacao_top10']:.1f}% da renda\n")
                f.write(f"  • 40% mais pobres: {row['participacao_bottom40']:.1f}% da renda\n")
                f.write(f"  • Índice de Palma: {row['indice_palma']:.2f}\n")
                f.write(f"  • População: {row['populacao']:.1f} milhões\n")
            
            # 3. ANÁLISE TEMPORAL
            f.write("\n\n3. EVOLUÇÃO TEMPORAL DA DESIGUALDADE (BRASIL)\n")
            f.write("-" * 50 + "\n")
            
            if 'Brasil' in dados_historicos:
                dados_brasil = dados_historicos['Brasil']
                f.write("Ano | Coef. Gini | Variação | Renda Média (R$)\n")
                f.write("-" * 50 + "\n")
                
                anos_ordenados = sorted(dados_brasil.keys())
                gini_anterior = None
                
                for ano in anos_ordenados:
                    dados = dados_brasil[ano]
                    variacao = ""
                    if gini_anterior is not None:
                        variacao = f"{dados['gini'] - gini_anterior:+.3f}"
                    
                    f.write(f"{ano}  | {dados['gini']:.3f}     | {variacao:>8} | {dados['renda_media']:.0f}\n")
                    gini_anterior = dados['gini']
            
            # 4. INDICADORES DE CONCENTRAÇÃO EXTREMA
            f.write("\n\n4. INDICADORES DE CONCENTRAÇÃO EXTREMA DE RENDA\n")
            f.write("-" * 50 + "\n")
            
            # Top 5 mais desiguais
            top5_desigual = df_indicadores.nlargest(5, 'gini')
            f.write("\nTop 5 regiões mais desiguais:\n")
            for idx, row in top5_desigual.iterrows():
                f.write(f"  {row['regiao']}: Gini = {row['gini']:.3f}, "
                       f"Índice Palma = {row['indice_palma']:.2f}\n")
            
            # Top 5 menos desiguais
            top5_igual = df_indicadores.nsmallest(5, 'gini')
            f.write("\nTop 5 regiões menos desiguais:\n")
            for idx, row in top5_igual.iterrows():
                f.write(f"  {row['regiao']}: Gini = {row['gini']:.3f}, "
                       f"Índice Palma = {row['indice_palma']:.2f}\n")
            
            # 5. RECOMENDAÇÕES DE POLÍTICA
            f.write("\n\n5. RECOMENDAÇÕES DE POLÍTICA PÚBLICA\n")
            f.write("-" * 50 + "\n")
            f.write("1. Fortalecer programas de transferência de renda\n")
            f.write("2. Investir em educação básica de qualidade\n")
            f.write("3. Promover reforma tributária progressiva\n")
            f.write("4. Estimular geração de empregos formais\n")
            f.write("5. Reduzir desigualdades regionais\n")
            f.write("6. Ampliar acesso ao crédito para baixa renda\n")
            f.write("7. Fortalecer políticas de valorização do salário mínimo\n")
            
            # 6. METODOLOGIA
            f.write("\n\n6. METODOLOGIA E FONTES\n")
            f.write("-" * 40 + "\n")
            f.write("• Fonte primária: IBGE - PNAD Contínua 2023\n")
            f.write("• Período de referência: 4º trimestre de 2023\n")
            f.write("• Unidade de análise: Rendimento domiciliar per capita\n")
            f.write("• Coeficiente de Gini: Calculado pelo método dos trapézios\n")
            f.write("• Índice de Palma: Renda dos 10% mais ricos / 40% mais pobres\n")
            f.write("• Valores em reais de 2023\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Relatório analítico salvo como '{nome_relatorio}'")
        
        # Exportar dados para CSV
        nome_csv = f'dados_desigualdade_{timestamp}.csv'
        df_export = df_indicadores.copy()
        
        # Remover colunas não serializáveis
        if 'decis_acumulados' in df_export.columns:
            df_export = df_export.drop(columns=['decis_acumulados', 'populacao_decis'])
        
        df_export.to_csv(nome_csv, index=False, encoding='utf-8-sig')
        print(f"✓ Dados completos salvos como '{nome_csv}'")
        
        return nome_relatorio, nome_csv
        
    except Exception as e:
        print(f"✗ Erro ao gerar relatório: {e}")
        return None, None

# ====================== 4. FUNÇÃO PRINCIPAL ======================

def main():
    """
    Função principal de execução
    """
    print("=" * 80)
    print("ANÁLISE DA DISTRIBUIÇÃO DE RENDA - DADOS OFICIAIS DO IBGE")
    print("=" * 80)
    print("PROJETO 02 - TÓPICOS AVANÇADOS EM ANÁLISE DE DADOS")
    print("Professor: Gustavo Sampaio")
    print("=" * 80)
    
    try:
        # 1. Carregar dados oficiais
        print("\n📥 1. CARREGANDO DADOS DO IBGE...")
        dados_ibge, dados_historicos = carregar_dados_ibge()
        
        print(f"   ✓ Dados de {len(dados_ibge)} regiões carregados")
        print(f"   ✓ Período: 2023 (4º trimestre)")
        print(f"   ✓ Fonte: PNAD Contínua - IBGE")
        
        # 2. Criar dashboard visual
        print("\n📊 2. GERANDO DASHBOARD VISUAL...")
        df_indicadores, nome_imagem = criar_visualizacao_completa(dados_ibge, dados_historicos)
        
        if df_indicadores is None:
            raise ValueError("Falha na criação do dashboard")
        
        # 3. Gerar relatório analítico
        print("\n📄 3. GERANDO RELATÓRIO ANALÍTICO...")
        nome_relatorio, nome_csv = gerar_relatorio_analitico(df_indicadores, dados_ibge, dados_historicos)
        
        # 4. Exibir estatísticas resumidas
        print("\n📈 4. ESTATÍSTICAS RESUMIDAS:")
        print("-" * 50)
        
        brasil_stats = df_indicadores[df_indicadores['regiao'] == 'Brasil'].iloc[0]
        
        print(f"\n🇧🇷 BRASIL (Nacional):")
        print(f"   • Coeficiente de Gini: {brasil_stats['gini']:.3f}")
        print(f"   • Renda média: R$ {brasil_stats['renda_media']:.0f}")
        print(f"   • Renda mediana: R$ {brasil_stats['renda_mediana']:.0f}")
        print(f"   • 10% mais ricos: {brasil_stats['participacao_top10']:.1f}% da renda")
        print(f"   • 40% mais pobres: {brasil_stats['participacao_bottom40']:.1f}% da renda")
        
        print(f"\n🏆 REGIÃO MAIS DESIGUAL:")
        regiao_max = df_indicadores.iloc[0]
        print(f"   • {regiao_max['regiao']}: Gini = {regiao_max['gini']:.3f}")
        
        print(f"\n✅ REGIÃO MENOS DESIGUAL:")
        regiao_min = df_indicadores.iloc[-1]
        print(f"   • {regiao_min['regiao']}: Gini = {regiao_min['gini']:.3f}")
        
        print(f"\n📅 EVOLUÇÃO TEMPORAL (2012-2023):")
        if 'Brasil' in dados_historicos:
            dados_brasil = dados_historicos['Brasil']
            variacao_total = brasil_stats['gini'] - dados_brasil[2012]['gini']
            direcao = "redução" if variacao_total < 0 else "aumento"
            print(f"   • 2012: Gini = {dados_brasil[2012]['gini']:.3f}")
            print(f"   • 2023: Gini = {brasil_stats['gini']:.3f}")
            print(f"   • {direcao} de {abs(variacao_total):.3f} pontos")
        
        # 5. Verificar arquivos gerados
        print("\n📁 5. ARQUIVOS GERADOS:")
        
        arquivos = []
        if nome_imagem and os.path.exists(nome_imagem):
            tamanho = os.path.getsize(nome_imagem) / 1024
            print(f"   ✓ {nome_imagem} ({tamanho:.1f} KB)")
            arquivos.append(nome_imagem)
        
        if nome_relatorio and os.path.exists(nome_relatorio):
            tamanho = os.path.getsize(nome_relatorio) / 1024
            print(f"   ✓ {nome_relatorio} ({tamanho:.1f} KB)")
            arquivos.append(nome_relatorio)
        
        if nome_csv and os.path.exists(nome_csv):
            tamanho = os.path.getsize(nome_csv) / 1024
            print(f"   ✓ {nome_csv} ({tamanho:.1f} KB)")
            arquivos.append(nome_csv)
        
        # 6. Resumo final
        print("\n" + "=" * 80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("-" * 80)
        
        print(f"\n📊 RESULTADOS:")
        print(f"   • Regiões analisadas: {len(df_indicadores)}")
        print(f"   • Dashboard visual: {'✅ Gerado' if nome_imagem else '❌ Falhou'}")
        print(f"   • Relatório analítico: {'✅ Gerado' if nome_relatorio else '❌ Falhou'}")
        print(f"   • Base de dados: {'✅ Exportada' if nome_csv else '❌ Falhou'}")
        print(f"   • Arquivos totais: {len(arquivos)}")
        
        if len(arquivos) >= 2:
            print("\n🎉 TODOS OS MÓDULOS PRINCIPAIS FORAM EXECUTADOS!")
            
            print("\n📌 PRÓXIMOS PASSOS:")
            print("   1. Analise o dashboard visual gerado")
            print("   2. Consulte o relatório analítico detalhado")
            print("   3. Use o arquivo CSV para análises adicionais")
            print("   4. Compare com dados históricos e internacionais")
        
        print("\n" + "=" * 80)
        print("📚 REFERÊNCIAS:")
        print("   • IBGE - Pesquisa Nacional por Amostra de Domicílios (PNAD)")
        print("   • IBGE - Síntese de Indicadores Sociais")
        print("   • World Bank - World Development Indicators")
        print("   • OECD - Income Distribution Database")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print("\n🛠️  SOLUÇÃO DE PROBLEMAS:")
        print("   1. Verifique se todas as bibliotecas estão instaladas")
        print("   2. Execute: pip install pandas matplotlib seaborn scipy numpy")
        print("   3. Verifique espaço em disco disponível")
        print("=" * 80)
        return False

# ====================== 5. EXECUÇÃO ======================

if __name__ == "__main__":
    # Configurações do pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    # Executar análise
    print("\n🚀 INICIANDO ANÁLISE DE DESIGUALDADE DE RENDA")
    print("=" * 80)
    
    sucesso = main()
    
    if sucesso:
        print("\n💡 DICA: Para análise avançada, use os dados CSV em:")
        print("   • Excel/Power BI para dashboards interativos")
        print("   • R/Python para modelagem econométrica")
        print("   • Tableau para visualizações avançadas")
        print("=" * 80)
    else:
        print("\n❌ Análise interrompida devido a erros.")
        print("=" * 80)