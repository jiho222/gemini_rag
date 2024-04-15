# 필수 모듈 임포트
from flask import Flask, render_template, request, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
from dotenv import load_dotenv

# 외부 서비스 및 유틸리티 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

# Flask 애플리케이션 설정
app = Flask(__name__)
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') # WTF를써서csrf공격이자동보호됨

# Flask 애플리케이션의 루트 디렉터리 내의 'uploads' 폴더에 대한 상대 경로 설정
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 폼 클래스 정의
class QuestionForm(FlaskForm):
    question = StringField('질문을 입력하세요', validators=[DataRequired()])
    submit = SubmitField('질문 제출')

# 뷰 함수 정의
@app.route('/', methods=['GET', 'POST'])
def index():
    form = QuestionForm()
    answer = None  # 답변을 저장할 변수 초기화

    if 'action' in request.form and request.form['action'] == 'delete_files':
        # uploads 폴더 내의 모든 파일 삭제 로직
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        for file in files:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        # 세션에서 PDF 경로 정보 삭제
        session.pop('pdf_path', None)
        # 파일 삭제 후 메시지 설정 (옵션)
        # flash('모든 업로드된 파일이 삭제되었습니다.')

    pdf_file = request.files.get('pdf')
    if pdf_file:
        filename = pdf_file.filename
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)
        session['pdf_path'] = pdf_path  # 파일 경로를 세션에 저장

    if form.validate_on_submit():
        question = form.question.data

        # 세션에 저장된 PDF 경로를 사용하여 질문 처리
        if 'pdf_path' in session:
            pdf_path = session['pdf_path']

            # PDF를 받고 페이지를 나누고 chunk로 나눔, 그것이 text
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(pages)

            # HuggingFace 임베딩 모델, 그것이 hf
            model_name = "jhgan/ko-sbert-nli"
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # docsearch = text를 hf로 임베딩한것을 Chroma 에 저장
            docsearch = Chroma.from_documents(texts, hf)

            # retriever 는 비정형 쿼리가 제공되면 문서를 반환하는 인터페이스이다. 문서를 저장하지 않고 반환만 한다.
            # retriever = docsearch(text+hf) + as retriever(mmr, k, fetch_k) 로 다합쳐서 질문
            retriever = docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k':3, 'fetch_k': 10}
            )
            # 'k':3, 'fetch_k': 10 는 가장 유사한 답변 10개 중에서 다양한 조합을 고려해서 3개를 뽑는다

            # 이제 다음의 명령어를 통해 PDF파일과 관련된 질문의 답변을 얻을 수 있다
            # relevant_documents = retriever.get_relevant_documents(question)

            # template를 이용해 prompt를 만든 후 RunnableMap을 이용해 chain으로 묶는다
            template = """
            Answer the question as based only on the following context: {context}
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

            chain = RunnableMap({
                # question과 유사한 문서를 뽑아낸 것을 context에 넣는다
                "context": lambda x: retriever.get_relevant_documents(x['question']),
                "question": lambda x: x['question']
            }) | prompt | gemini

            answer = chain.invoke({'question': question}).content

    return render_template('index.html', form=form, answer=answer)

# # 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)