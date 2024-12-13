#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '16AuPtqc6sh0wZBKqcUMAbEkwHAe2S5Cd'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/j39jBPc/common.jpg",
            "https://i.ibb.co/j5879vN/common-1.jpg",
            "https://i.ibb.co/7K5B48L/common-2.jpg"
        ],
        'videos': [
            "https://youtu.be/srvL8tPKlo4?feature=shared",
            "https://youtu.be/Ra0y1PjUe-Y?feature=shared",
            "https://youtu.be/PQjqxJgXqyU?feature=shared"
        ],
        'texts': [
            "작품1 밥 잘 사주는 예쁜 누나 '그냥 아는 사이'로 지내던 두 남녀가 사랑에 빠지면서 그려가게 될 '진짜 연애'에 대한 이야기",
            "작품2 그 해 우리는 함께해서 더러웠고 다신 보지 말자!로 끝났어야 할 인연이 10년이 흘러 카메라 앞에 강제 소환 되어 펼쳐지는 청춘 다큐를 가장한 아찔한 로맨스 드라마",
            "작품3 멜로가 체질 서른 살 여자 친구들의 고민, 연애, 일상을 그린 코믹 드라마"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/NNYDV2Q/common.jpg",
            "https://i.ibb.co/Q9ngtZ1/common-3.jpg",
            "https://i.ibb.co/D5GYn2Q/common-2.jpg"
        ],
        'videos': [
            "https://youtu.be/Lwr0nn0Hcps?feature=shared",
            "https://youtu.be/P3F_ydSlTMs?feature=shared",
            "https://youtu.be/TQUsLhMMVa0?feature=shared"
        ],
        'texts': [
            "작품1 사랑의 불시착 어느 날 돌풍과 함께 패러글라이딩 사고로 북한에 불시착한 재벌 상속녀 윤세리와 그녀를 숨기고 지키다 사랑하게 되는 특급 장교 리정혁의 절대 극비 러브스토리를 그린 드라마",
            "작품2 THE K2 전쟁 용병 출신의 보디가드 K2와 그를 고용한 대선 후보의 아내, 그리고 세상과 떨어져 사는 소녀! 로열패밀리를 둘러싼 은밀하고 강렬한 보디가드 액션 드라마",
            "작품3 복수가 돌아왔다 학교 폭력 가해자로 몰려 부당하게 퇴학을 당한 강복수가, 어른이 돼 학교로 다시 돌아가 복수를 계획하지만, 복수는 고사하고 또다시 사건에 휘말리고 사랑도 다시 하는 엉뚱하면서 따뜻한 감성 로맨스 드라마"
        ]
    }

}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

