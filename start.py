import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def run_model(script_name):
    print(f"\nExecutando {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    return result.stdout

def extract_results(output):
    # Extrair métricas usando regex
    accuracy = float(re.search(r"Acurácia: (\d+\.\d+)", output).group(1))
    precision = float(re.search(r"Precisão \(média\): (\d+\.\d+)", output).group(1))
    recall = float(re.search(r"Recall \(média\): (\d+\.\d+)", output).group(1))
    f1 = float(re.search(r"F1-score \(média\): (\d+\.\d+)", output).group(1))
    time = float(re.search(r"Tempo para treino do modelo:\n(\d+\.\d+)", output).group(1))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': time
    }

def plot_comparison(results):
    models = ['SVM', 'Random Forest', 'Regressão Logística']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Criar figura com 3 subplots
    plt.figure(figsize=(15, 12))
    
    # Gráfico 1: Métricas de desempenho
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    
    ax1 = plt.subplot(3, 1, 1)
    bar_width = 0.25
    index = np.arange(len(metrics))
    
    for i, (model_name, model) in enumerate(results.items()):
        values = [model[metric] for metric in metrics]
        bars = ax1.bar(index + i * bar_width, values, bar_width, 
                      label=models[i], color=colors[i])
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('Comparação de Métricas de Desempenho', fontsize=14, pad=15)
    ax1.set_xticks(index + bar_width)
    ax1.set_xticklabels(metric_names, fontsize=11)
    ax1.set_ylabel('Valor', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico 2: Tempo de execução (barras verticais)
    ax2 = plt.subplot(3, 1, 2)
    times = [result['time'] for result in results.values()]
    bars = ax2.bar(models, times, color=colors)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=11)
    
    ax2.set_title('Tempo de Execução (Barras Verticais)', fontsize=14, pad=15)
    ax2.set_ylabel('Tempo (segundos)', fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico 3: Tempo de execução (barras horizontais para melhor visualização)
    ax3 = plt.subplot(3, 1, 3)
    bars = ax3.barh(models, times, color=colors)
    
    # Adicionar valores nas barras
    for bar in bars:
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.4f}s',
                ha='left', va='center', fontsize=11)
    
    ax3.set_title('Tempo de Execução (Barras Horizontais)', fontsize=14, pad=15)
    ax3.set_xlabel('Tempo (segundos)', fontsize=12)
    ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir tabela resumo
    print("\nResumo dos Resultados:")
    print(f"{'Modelo':<20} {'Acurácia':<10} {'Precisão':<10} {'Recall':<10} {'F1-Score':<10} {'Tempo (s)':<10}")
    print("-" * 70)
    for model_name, result in zip(models, results.values()):
        print(f"{model_name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['time']:<10.4f}")

if __name__ == "__main__":
    print("Executando todos os modelos e gerando gráfico comparativo...")
    
    # Executar modelos e capturar saída
    svm_output = run_model("support_vector_machine.py")
    rf_output = run_model("random_forest.py")
    logreg_output = run_model("regressao_logistica.py")
    
    # Extrair resultados
    results = {
        'svm': extract_results(svm_output),
        'rf': extract_results(rf_output),
        'logreg': extract_results(logreg_output)
    }
    
    # Gerar gráfico
    plot_comparison(results)
    
    print("\nProcesso concluído. Verifique o gráfico 'comparacao_modelos.png'")