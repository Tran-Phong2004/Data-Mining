from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép tất cả các yêu cầu từ các nguồn khác

# Load mô hình đã huấn luyện
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        age = int(request.form.get('age'))  # Tuổi khách hàng
        gender = int(request.form.get('gender'))  # Giới tính (0: Nam, 1: Nữ)
        income = float(request.form.get('income'))  # Thu nhập hàng năm
        purchases = int(request.form.get('purchases'))  # Số lần mua
        category = int(request.form.get('category'))  # Danh mục sản phẩm (0-4)
        time_spent = int(request.form.get('time_spent'))  # Thời gian trên website
        loyalty_program = int(request.form.get('loyalty_program'))  # Thành viên chương trình khách hàng thân thiết
        discounts = int(request.form.get('discounts'))  # Số lần sử dụng giảm giá

        # Tạo dữ liệu đầu vào cho mô hình dưới dạng numpy array
        input_data = np.array([age, gender, income, purchases, category, time_spent, loyalty_program, discounts]).reshape(1, -1)

        # Dự đoán với mô hình đã load
        prediction = model.predict(input_data)
        #print(prediction)
        # Trả về kết quả dự đoán dưới dạng JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()