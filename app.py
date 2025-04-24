from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
model = load_model('my_model.keras')
classes = {
    1: 'Giới hạn tốc độ (20km/h)',
    2: 'Giới hạn tốc độ (30km/h)',
    3: 'Giới hạn tốc độ (50km/h)',
    4: 'Giới hạn tốc độ (60km/h)',
    5: 'Giới hạn tốc độ (70km/h)',
    6: 'Giới hạn tốc độ (80km/h)',
    7: 'Hết giới hạn tốc độ (80km/h)',
    8: 'Giới hạn tốc độ (100km/h)',
    9: 'Giới hạn tốc độ (120km/h)',
    10: 'Cấm vượt',
    11: 'Cấm vượt với xe trên 3.5 tấn',
    12: 'Nhường đường tại ngã tư',
    13: 'Đường ưu tiên',
    14: 'Nhường đường',
    15: 'Dừng lại',
    16: 'Cấm tất cả các phương tiện',
    17: 'Cấm xe trên 3.5 tấn',
    18: 'Cấm vào',
    19: 'Cảnh báo chung',
    20: 'Đường cong nguy hiểm bên trái',
    21: 'Đường cong nguy hiểm bên phải',
    22: 'Đường cong đôi',
    23: 'Đường gồ ghề',
    24: 'Đường trơn',
    25: 'Đường hẹp bên phải',
    26: 'Đang thi công',
    27: 'Đèn tín hiệu giao thông',
    28: 'Khu vực người đi bộ',
    29: 'Trẻ em qua đường',
    30: 'Xe đạp băng qua',
    31: 'Cẩn thận băng/tuyết',
    32: 'Động vật hoang dã băng qua',
    33: 'Hết giới hạn tốc độ và lệnh cấm vượt',
    34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước',
    36: 'Chỉ được đi thẳng',
    37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái',
    39: 'Giữ bên phải',
    40: 'Giữ bên trái',
    41: 'Vòng xuyến bắt buộc',
    42: 'Hết lệnh cấm vượt',
    43: 'Hết lệnh cấm vượt với xe trên 3.5 tấn'
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        img = load_img(BytesIO(file.read()), target_size=(30, 30, 3))
        img = img_to_array(img)
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
