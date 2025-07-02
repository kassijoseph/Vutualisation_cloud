# Utilise une image Python officielle comme base
FROM python:3.11-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers requirements.txt et installe les dépendances
COPY requirements.txt .
# installer les dépendances python
# utiliser --no-chache-dir pour éviter de stocker les paquets dans le cache
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le contenu local dans le dossier /app du conteneur
COPY . .

# Expose le port 5000
EXPOSE 5000

# Commande pour lancer l’application Flask
CMD ["python", "app.py"]
 