import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import time
import timeit

# Carregar o dataset com encoding ISO-8859-1
df = pd.read_csv('spotify_dataset.csv', encoding='ISO-8859-1')

# Pré-processamento dos dados
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

# Definir target (popular se Popularity >= 80)
df['Popularity_Target'] = (df['Spotify Popularity'] >= 80).astype(int)

# Selecionar features
features = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach',
            'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes',
            'TikTok Views', 'Explicit Track']

# Lidar com valores ausentes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[features])
y = df['Popularity_Target']

def train_model():
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train, y_train)
    
    training_time = timeit.timeit(train_model, number=10)/10


# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Medição de tempo de treino
start_time = time.time()
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Fazer previsões
y_pred = logreg_model.predict(X_test)

# Avaliação
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Resultados formatados conforme solicitado
print("\nPerformance média do modelo:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão (média): {report['macro avg']['precision']:.2f}")
print(f"Recall (média): {report['macro avg']['recall']:.2f}")
print(f"F1-score (média): {report['macro avg']['f1-score']:.2f}")

print("\nTempo para treino do modelo:")
print(f"{max(training_time, 0.01):.2f} segundos") 

# Relatório completo (opcional)
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))