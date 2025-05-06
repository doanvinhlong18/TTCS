from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
model = load_model('my_model.keras')
classes = {
    1: 'Tốc độ tối đa cho phép (20km/h)',
    2: 'Tốc độ tối đa cho phép (30km/h)',
    3: 'Tốc độ tối đa cho phép (50km/h)',
    4: 'Tốc độ tối đa cho phép (60km/h)',
    5: 'Tốc độ tối đa cho phép (70km/h)',
    6: 'Tốc độ tối đa cho phép (80km/h)',
    7: 'Hết giới hạn tốc độ (80km/h)',
    8: 'Tốc độ tối đa cho phép (100km/h)',
    9: 'Tốc độ tối đa cho phép (120km/h)',
    10: 'Cấm vượt',
    11: 'Cấm xe tải vượt',
    12: 'Giao nhau với đường không ưu tiên',
    13: 'Đường ưu tiên',
    14: 'Giao nhau với đường ưu tiên',
    15: 'Dừng lại',
    16: 'Đường cấm',
    17: 'Cấm xe tải có trọng lượng vượt quá giới hạn cho phép',
    18: 'Đường một chiều',
    19: 'Cảnh báo nguy hiểm',
    20: 'Chỗ ngoặt nguy hiểm bên trái',
    21: 'Chỗ ngoặt nguy hiểm bên phải',
    22: 'Nhiều chỗ ngoặt nguy hiểm liên tiếp',
    23: 'Đường lồi lõm',
    24: 'Đường trơn',
    25: 'Đường bị thu hẹp bên phải',
    26: 'Công trường',
    27: 'Giao nhau có tín hiệu đèn',
    28: 'Đường dành cho người đi bộ cắt ngang',
    29: 'Trẻ em qua đường',
    30: 'Đường người đi xe đạp cắt ngang',
    31: 'Cẩn thận băng/tuyết',
    32: 'Động vật hoang dã băng qua',
    33: 'Hết tất cả lệnh cấm',
    34: 'Các xe chỉ đuợc rẽ phải',
    35: 'Các xe chỉ đuợc rẽ trái',
    36: 'Hướng đi thẳng phải theo',
    37: 'Các xe chỉ đuợc đi thẳng và rẽ phải',
    38: 'Các xe chỉ đuợc đi thẳng và rẽ trái',
    39: 'Hướng phải đi vòng sang phải',
    40: 'Hướng phải đi vòng sang trái',
    41: 'Nơi giao nhau chạy theo vòng xuyến',
    42: 'Hết cấm vượt',
    43: 'Hết lệnh cấm vượt với xe trên 3.5 tấn'
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        img = load_img(BytesIO(file.read()), target_size=(30, 30))
        img = img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        label = classes.get(class_idx + 1, "Không xác định")

        return jsonify({
            'class_index': int(class_idx),
            'label': label
        })
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
