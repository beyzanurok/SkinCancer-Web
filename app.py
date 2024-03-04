from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Model ve sınıfların yüklenmesi
model = load_model('vgg16_finetuned_model.h5')  # Model dosyanızın adı
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Sınıf etiketleri

@app.route("/", methods=["GET", "POST"])
def runhome():
    return render_template("home.html")

@app.route("/showresult", methods=["GET", "POST"])
def show():
    if request.method == "POST":
        pic = request.files["pic"]
        if pic:
            inputimg = Image.open(pic).convert('RGB')  # RGB formatına dönüştürme
            inputimg = inputimg.resize((75, 100))  # Modelin beklediği boyutlara uygun olarak yeniden boyutlandırma
            img = np.array(inputimg)
            img = np.expand_dims(img, axis=0)  # Modelin beklediği şekle getirme
            img = img / 255.0  # Normalizasyon
            
            result = model.predict(img)
            class_id = np.argmax(result, axis=1)[0]
            result_text = classes[class_id]

            # Sınıf indeksine göre bilgilendirme metni
            info_texts = {
                0: "Aktinik keratoz, uzun süreli kontrolsüz güneşe maruz kalmaya bağlı olarak en çok güneş gören bölgelerde görülen deride anormal hücre gelişimini yansıtan deri değişiklikleridir. En sık yüzeyi pürtüklü yama şeklinde görülürler. Aktinik keratozların düşük oranda deri kanserine dönüşme riski vardır.",
                1: "Bazal hücreli karsinom, bir cilt kanseri türüdür. BHK olarak kısaltılır. Deride bulunan eski hücrelerin ölmesiyle onların yerine yenilerini üreten, bazal hücre olarak adlandırılan hücre tipinde başlar. BHK farklı şekillerde olabilmekle birlikte sıklıkla ciltte hafif şeffaf bir yumru olarak belirir.",
                2: "Benign likenoid keratoz, liken planus benzeri keratoz olarak da bilinen, derinin malign lezyonlarıyla karıştırılabilen sık bir anti- tedir. Genellikle 35-65 yaş arası kadınlarda, yüz ve üst gövdede, asemptomatik soliter pembe-kırmızı-kahverengi papül veya hafif endüre plak olarak izlenir.",
                3: "Dermatofibromlar orta yaş erişkinlerde oldukça sık görülen, fibroblastlar ve histiyositlerden kaynaklanan benign tümörlerdir. Genellikle alt ekstremitede yerleşen sert, tek veya multipl, papül, plak ya da nodül şeklindeki lezyonlarla karakterizedir. ",
                4: "Melanotik nevus (Nevus pigmentosus; Pigmentli nevus; Ben), insanların büyük bölümünde bir ya da birkaç tane olabilen, melanin üreten hücrelerin (melanosit) iyi huylu tümörüdür.",
                5: "Piyojenik granülomlar küçük, yuvarlak ve genellikle kanlı kırmızı renkli cilt büyümeleridir. Çok sayıda kan damarı içerdikleri için kanamaya eğilimlidirler. Ayrıca lobüler kılcal hemanjiyom veya granüloma telenjiektatikum olarak da bilinirler.",
                6: "Cilt kanserinin en ciddi türü olan melanom, cildinize rengini veren pigment olan melanin üreten hücrelerde (melanositler) gelişir. Melanom ayrıca gözlerinizde ve nadiren vücudunuzun içinde (burnunuz veya boğazınız gibi) da oluşabilir. Tüm melanomların kesin nedeni açık değildir, ancak güneş ışığından veya bronzlaşma lambalarından ve ultraviyole (UV) radyasyona maruz kalmak melanom geliştirme riskinizi artırır."
                

            }

            info = info_texts.get(class_id, "Bilgi bulunamadı.")
            return render_template("result.html", result=result_text, info=info)
    # POST olmayan istekler için ana sayfaya yönlendir
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)


