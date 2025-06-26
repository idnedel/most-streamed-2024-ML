import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import time

# Carregar o dataset com encoding ISO-8859-1
df = pd.read_csv('spotify_dataset.csv', encoding='ISO-8859-1')

# Pré-processamento dos dados
# Converter colunas numéricas que estão como strings
def convert_to_float(x):
    if isinstance(x, str):
        return float(x.replace('"', '').replace(',', ''))
    return float(x)

numeric_cols = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 
                'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 
                'TikTok Views', 'YouTube Playlist Reach', 'Apple Music Playlist Count',
                'AirPlay Spins', 'SiriusXM Spins', 'Deezer Playlist Count',
                'Deezer Playlist Reach', 'Amazon Playlist Count', 'Pandora Streams',
                'Pandora Track Stations', 'Soundcloud Streams', 'Shazam Counts',
                'TIDAL Popularity']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_float)

# Definir a popularidade como alvo (binarizado)
# Consideraremos popular se Popularity >= 80 (valor arbitrário baseado na distribuição)
df['Popularity_Target'] = (df['Spotify Popularity'] >= 80).astype(int)

# Selecionar features relevantes
features = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach',
            'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes',
            'TikTok Views', 'Explicit Track']

# Lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[features])
y = df['Popularity_Target']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo SVM
start_time = time.time()
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Fazer previsões
y_pred = svm_model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Resultados
print("\na. Performance média do modelo:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão (média): {report['macro avg']['precision']:.2f}")
print(f"Recall (média): {report['macro avg']['recall']:.2f}")
print(f"F1-score (média): {report['macro avg']['f1-score']:.2f}")

print("\nb. Tempo para treino do modelo:")
print(f"{training_time:.2f} segundos")

# Mostrar relatório completo
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))