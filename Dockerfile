FROM python:3.10

# Node.js kurulumu (Vite/React arayüzünü derlemek için)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Proje dosyalarını kopyala
COPY . /app

# Python bağımlılıklarını kur (all flagi ile faiss, openai vs.)
RUN pip install --no-cache-dir -e .[all]
RUN pip install --no-cache-dir uvicorn fastapi

# UI arayüzünü derle
WORKDIR /app/ui
RUN npm install
RUN npm run build

# Çalışma dizinini ana klasöre geri döndür
WORKDIR /app

# Hugging Face Spaces varsayılan olarak 7860 portunu dinler
ENV INTENTMIND_API_PORT=7860
ENV INTENTMIND_CORS_ORIGINS="*"
# Hafızanın boş gelmemesi için dummy bir örnek veri de oluşturabiliriz, 
# ama Hugging Face kendi persistent storage alanını kullanır.

# Uygulamayı başlat
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
