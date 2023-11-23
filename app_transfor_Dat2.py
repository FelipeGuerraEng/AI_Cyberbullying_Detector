import dash
import urllib.parse
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from flask import Flask
from dash import Dash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from joblib import load
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345@localhost:5432/tesis'
db = SQLAlchemy(app)


app.config['SECRET_KEY'] = 'a3c2ab887b59a08da4464d3bb22780c3a5b2c925bfbe6e5b' # provsional

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)  # id autoincrementado
    username = db.Column(db.String(100), unique=True)  # nuevo campo username
    password = db.Column(db.String(512), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password, password)

@login_manager.user_loader
def user_loader(username):
    return User.query.get(username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('upload_file'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        return redirect(url_for('upload_file'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Input error')
            return redirect(url_for('register'))

        user = User.query.filter_by(username=username).first()
        if user:
            flash('User already exists')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Define el modelo de tu tabla
class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    autor = db.Column(db.String(120), nullable=False)
    comentario = db.Column(db.Text, nullable=False)
    genero = db.Column(db.String(120), nullable=False)
    orientacion_sexual = db.Column(db.String(120), nullable=False)
    grupo_etareo = db.Column(db.String(120), nullable=False)
    estrato = db.Column(db.String(120), nullable=False)
    discapacidad = db.Column(db.String(120), nullable=False)
    victima_del_conflicto = db.Column(db.String(120), nullable=False)
    red_de_apoyo = db.Column(db.String(120), nullable=False)
    
# Creación de las tablas
with app.app_context():
    db.create_all()

# Crea el modelo
model_name = 'FelipeGuerra/colombian-spanish-cyberbullying-detector'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    if request.method == 'POST':
        autor = request.form.get('autor')
        comentario = request.form.get('comentario')
        genero = request.form.get('genero')
        orientacion_sexual = request.form.get('orientacion_sexual')
        grupo_etareo = request.form.get('grupo_etareo')
        estrato = request.form.get('estrato')
        discapacidad = request.form.get('discapacidad')
        victima_del_conflicto = request.form.get('victima_del_conflicto')
        red_de_apoyo = request.form.get('red_de_apoyo')

        new_tweet = Tweet(autor=autor, comentario=comentario, genero=genero, orientacion_sexual=orientacion_sexual,
                          grupo_etareo=grupo_etareo, estrato=estrato, discapacidad=discapacidad,
                          victima_del_conflicto=victima_del_conflicto, red_de_apoyo=red_de_apoyo)
        db.session.add(new_tweet)
        db.session.commit()

        return render_template('index.html', tweets=Tweet.query.all())

    elif request.method == 'GET':
        return render_template('index.html', tweets=Tweet.query.all())
        
@app.route('/delete/<int:id>')
def delete_tweet(id):
    tweet_to_delete = Tweet.query.get_or_404(id)
    db.session.delete(tweet_to_delete)
    db.session.commit()
    return redirect('/')
    
@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_tweet(id):
    tweet_to_edit = Tweet.query.get_or_404(id)
    if request.method == 'POST':
        autor = request.form.get('autor')
        comentario = request.form.get('comentario')
        genero = request.form.get('genero')
        orientacion_sexual = request.form.get('orientacion_sexual')
        grupo_etareo = request.form.get('grupo_etareo')
        estrato = request.form.get('estrato')
        discapacidad = request.form.get('discapacidad')
        victima_del_conflicto = request.form.get('victima_del_conflicto')
        red_de_apoyo = request.form.get('red_de_apoyo')

        edited_tweet = Tweet(autor=autor, comentario=comentario, genero=genero, orientacion_sexual=orientacion_sexual,
                             grupo_etareo=grupo_etareo, estrato=estrato, discapacidad=discapacidad,
                             victima_del_conflicto=victima_del_conflicto, red_de_apoyo=red_de_apoyo)
        db.session.delete(tweet_to_edit)
        db.session.add(edited_tweet)
        db.session.commit()
        return redirect(url_for('upload_file'))
    return render_template('index.html', tweets=Tweet.query.all(), tweet_to_edit=tweet_to_edit)

@app.route('/analizar', methods=['GET'])
def analizar():
    tweets = Tweet.query.all()
    df = pd.DataFrame([(tweet.autor, tweet.comentario, tweet.genero, tweet.orientacion_sexual, tweet.grupo_etareo,
                        tweet.estrato, tweet.discapacidad, tweet.victima_del_conflicto, tweet.red_de_apoyo)
                       for tweet in tweets],
                      columns=['autor', 'comentario', 'genero', 'orientacion_sexual', 'grupo_etareo', 'estrato',
                               'discapacidad', 'victima_del_conflicto', 'red_de_apoyo'])
                                                           
    df_copy = df.copy()  
    df['comentario'] = df['comentario'].str.replace('\r\n', ' ').str.strip()

    results = []
    for comentario in df['comentario']:
        # Tokeniza el texto
        inputs = tokenizer(comentario, return_tensors='pt')

        # Realiza la predicción
        outputs = model(**inputs)

        # Calcula las probabilidades con la función softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Obtiene la clase predicha
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        # Almacena las probabilidades y la clase predicha
        results.append({'label': 'Bullying' if predicted_class == 1 else 'Not_bullying', 
                        'score': probabilities[0, predicted_class].item()})

    # Extrae las etiquetas de las predicciones
    df['ciberacoso'] = [result['label'] for result in results]
    df['ciberacoso'] = df['ciberacoso'].map({'Not_bullying': 'No', 'Bullying': 'Sí'})  # Ajusta según tus etiquetas reales
    df_copy['ciberacoso'] = df['ciberacoso'] 
    df_copy.to_csv('datos.csv', index=False)
    
    return redirect('/visualizacion')  
    
@app.route('/analizar_csv',  methods=['GET', 'POST'])
def analizar_csv():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            df = pd.read_csv(file)
            
    df_copy = df.copy()
    
    df['comentario'] = df['comentario'].str.replace('\r\n', ' ').str.strip()

    results = []
    for comentario in df['comentario']:
        # Tokeniza el texto
        inputs = tokenizer(comentario, return_tensors='pt')

        # Realiza la predicción
        outputs = model(**inputs)

        # Calcula las probabilidades con la función softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Obtiene la clase predicha
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        # Almacena las probabilidades y la clase predicha
        results.append({'label': 'Bullying' if predicted_class == 1 else 'Not_bullying', 
                        'score': probabilities[0, predicted_class].item()})

    # Extrae las etiquetas de las predicciones
    df['ciberacoso'] = [result['label'] for result in results]
    df['ciberacoso'] = df['ciberacoso'].map({'Not_bullying': 'No', 'Bullying': 'Sí'})  # Ajusta según tus etiquetas reales
    df_copy['ciberacoso'] = df['ciberacoso'] 
    df_copy.to_csv('datos.csv', index=False)
    return redirect('/visualizacion')

dash_app = Dash(__name__, server=app, routes_pathname_prefix='/visualizacion/')

dash_app.layout = html.Div([

   html.Div(
        children=[
            html.Div(
                children=[
                    html.A("AI Cyberbullying Detector", href="/", style={'color': 'white', 'fontSize': 22, 'textDecoration': 'none', 'fontFamily': 'Arial'}),
                ],
                className='six columns',
                style={'paddingLeft': '10px'}
            ),
            html.Div(
                children=[
                    html.Ul(
                        children=[
                            html.Li(
                                children=[html.A("Home", href="/", style={'color': 'gray', 'fontSize': 18, 'textDecoration': 'none', 'fontFamily': 'Arial'})],
                                style={'display': 'inline-block', 'marginLeft': '980px'}
                            ),
                            html.Li(
                                children=[html.A("Cerrar sesión", href="/logout", style={'color': 'gray', 'fontSize': 18, 'textDecoration': 'none', 'fontFamily': 'Arial'})],
                                style={'display': 'inline-block', 'marginLeft': '60px'}
                            ),
                        ],
                        style={'listStyleType': 'none', 'margin': 0, 'padding': 0, 'overflow': 'hidden'}
                    )
                ],
                className='six columns',
                style={'textAlign': 'right', 'paddingRight': '10px'}
            ),
        ],
        style={'backgroundColor': '#343a40', 'padding': '10px', 'marginBottom': '10px', 'display': 'flex'}
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Dropdown(
                        id='columna1',
                        options=[{'label': i.replace("_", " ").capitalize(), 'value': i} for i in
                                 ['genero', 'orientacion_sexual', 'grupo_etareo', 'estrato', 'discapacidad',
                                  'victima_del_conflicto', 'red_de_apoyo']],
                        value='genero'
                    ),
                    dcc.Graph(id='grafico1'),
                ],
                style={'width': '50%', 'display': 'inline-block'},
            ),
            html.Div(
                children=[
                    dcc.Dropdown(
                        id='columna2',
                        options=[{'label': i.replace("_", " ").capitalize(), 'value': i} for i in
                                 ['genero', 'orientacion_sexual', 'grupo_etareo', 'estrato', 'discapacidad',
                                  'victima_del_conflicto', 'red_de_apoyo']],
                        value='estrato'
                    ),
                    dcc.Graph(id='grafico2'),
                ],
                style={'width': '50%', 'display': 'inline-block'},
            ),
        ],
    ),
    dash_table.DataTable(sort_action='native', filter_action='native', column_selectable='single',
        id='table',
        page_action='native',
        page_size=10,
        columns=[{"name": "Usuario" if i == "autor" else i.replace("_", " ").capitalize(), "id": i} for i in
                 ['autor', 'comentario', 'genero', 'orientacion_sexual', 'grupo_etareo', 'estrato', 'discapacidad',
                  'victima_del_conflicto', 'red_de_apoyo', 'ciberacoso']],
        style_cell={'whiteSpace': 'normal', 'height': 'auto', 'textAlign': 'center'},  # Justifica el texto a la izquierda
        style_table={'overflowX': 'auto', 'marginTop': '20px'}  # Añade un scroll horizontal para una visualización más compacta
    ),
    dbc.Toast(
        "La tabla se exportó al archivo datos.csv.",
        id="toast",
        is_open=False,
        duration=4000,
        style={"font-size": "16px", "margin-top": "10px"}  # Estilo directo aplicado al aviso
    ),
    html.Div(
        id="export-button-container",
        className="text-center",
        children=[
            dbc.Button('Exportar a CSV', id='btn_export', color='primary', className="mt-4"),
            html.P("AI Cyberbullying Detector - Derechos Reservados", style={'fontSize': 10, 'textAlign': 'right'})
        ],
    ),
    dcc.Download(id='download')
])

@dash_app.callback(
    Output('toast', 'is_open'),
    Input('btn_export', 'n_clicks'),
    prevent_initial_call=True
)
def export_to_csv(n_clicks):
    if n_clicks:
        return True
    return False


@dash_app.callback(
    Output('grafico1', 'figure'),
    [Input('columna1', 'value')]
)
def update_graph1(columna):
    df = pd.read_csv('datos.csv')
    return px.histogram(df, x=columna, color='ciberacoso', labels={columna: columna.replace("_", " ").capitalize()})

    
@dash_app.callback(
    Output('grafico2', 'figure'),
    [Input('columna2', 'value')]
)
def update_graph2(columna):
    df = pd.read_csv('datos.csv')
    return px.histogram(df, x=columna, color='ciberacoso', labels={columna: columna.replace("_", " ").capitalize()})


@dash_app.callback(
    Output('table', 'data'),
    [Input('columna1', 'value'), Input('columna2', 'value')]
)
def update_table(columna1, columna2):
    df = pd.read_csv('datos.csv')  # Lee los datos del archivo CSV
    return df.to_dict('records')
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
