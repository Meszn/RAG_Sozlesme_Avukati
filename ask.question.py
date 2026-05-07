"""
Problem tanimi: sozlesme asistani 
    - Kullanicinin yukledigi sozlesme dosyasından icerik cıkarmak
    - bu icerigi vektorel olarak temsil etmek
    - faiss kullanarak bu vektorel temsili saklamak ve sorgulamak
    - kullanicinin sorularini al, sonra git dbden ilgili bilgiyi cikar ve gpt 3.5 ile cevapla

Kullanilan teknolojiler:
    - embedding: metni vektorel olarak temsil etmek icin kullanilir
    - faiss: vektorel temsilleri saklamak ve sorgulamak icin kullanilir. hızlı benzerlik aramasi yapar.
    - gpt 3.5: kullanicinin sorularini cevaplamak icin

RAG:
-Retrieval Augmented Generation, kullanicinin sorularini cevaplamak icin hem retrieval (bilgi cekme) hem de generation (cevap uretme) tekniklerini kullanir.
- Dil modellerine bilgi destegi saglayn bir tekniktir.
- Retrieval: kullanicinin sorusu embedding olarak temsil edilir ve faiss kullanarak en benzer vektorel temsiller bulunur. 
        Bu temsiller, kullanicinin sorusuna en uygun bilgiyi saglar.
- augmentation: retrieval asamasinda bulunan benzer vektorel temsiller, kullanicinin sorusunu cevaplamak icin gpt 3.5'e girdi olarak verilir. 
        Bu sayede, gpt 3.5 sadece kullanicinin sorusunu degil, ayni zamanda ilgili bilgiyi de kullanarak daha dogru ve anlamli cevaplar uretebilir.
- generation: gpt 3.5, kullanicinin sorusunu ve retrieval asamasinda bulunan benzer vektorel temsilleri kullanarak cevap uretir.
        -1) tarih
        -2) ücret
        -3) taraflar

Neden RAG kullanilir?
- RAG, kullanicinin sorularini cevaplamak icin hem retrieval (bilgi cekme) hem de generation (cevap uretme) tekniklerini kullanarak daha dogru ve anlamli cevaplar uretebilir.
- RAG, dil modellerine bilgi destegi saglayarak, kullanicinin sorularina daha spesifik ve dogru cevaplar uretmesini saglar.

"""
#import libraries 
import os 
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from openai import OpenAI


#.env dosyasindan ortam degiskenlerini yükle 
load_dotenv()

#openai api anahtarinin alinmasi 
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

#Sentence transformers modelimiz, kucuk ve hizli embedding modeli
model = SentenceTransformer("BAAI/bge-m3")

#faiss index dosyasini yukle(onceden olusturulmus vektor veritabanimiz)
index = faiss.read_index(".\data\contracy_index.faiss")

#Chunklanmış metin verisini yükle 
with open("data/contracy_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

#kullanicidan gelen sorulari al
while True:
    
    #kullanicidan soru al
    question = input("\n Sorunuzu girniz:")

    #cıkmak icin 
    if question.lower() in ["exit","quit", "q"]:
        print("Çıkış yapılıyor...")
        break
    
    #kullanicinin sorularini verktöre cevirelim
    question_embedding = model.encode([question])

    #faiss veri tabanindan en yakin 3 chunki ara 
    k = 3
    distances, indices = index.search(np.array(question_embedding), k)

    # bulunan chunkları birleştirerek baglam yani context olustur
    retrived_chunks = [chunks[i] for i in indices[0]]
    context = "\n ---------------\n".join(retrived_chunks)

    # llm'e gönderilecek sistem promptu 
    prompt = f"""
                Sen Türkçe çalışan profesyonel bir sözleşme analiz asistanısın.

                Kurallar:
                - Kullanıcının sorusuna SADECE sorulan şeyi cevapla.
                - Gereksiz sözleşme özeti üretme.
                - Gereksiz risk analizi üretme.
                - Gereksiz başlıklar oluşturma.
                - Kısa, net ve profesyonel cevap ver.
                - Maksimum 3-5 cümle kullan.
                - Eğer bilgi sözleşmede yoksa açıkça belirt.
                - Bilgi uydurma.
                - Sadece verilen context üzerinden konuş.
                - Hukuki kesinlik ifade etme.
                - Gereksiz açıklama yapma.
                - Kullanıcı istemedikçe tam analiz yapma.

                Önemli:
                Eğer kullanıcı kısa ve spesifik soru soruyorsa:
                - yalnızca ilgili maddeyi açıkla
                - doğrudan cevabı ver
                - ekstra detay ekleme

                context :
                {context}

                Question :
                {question}

                Answer :

"""
    response = client.chat.completions.create(

        model = "gpt-3.5-turbo",
        messages = [{"role":"user", "content":prompt}],
        temperature=0.2 # daha kararli cevalar icin dusuk deger
    )

    print("AI Assistant: \n", response.choices[0].message.content.strip())


