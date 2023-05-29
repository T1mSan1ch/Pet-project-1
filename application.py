from joblib import load
from dash import Dash, html, dcc, Input, Output, callback_context
import pandas as pd
import numpy as np


app = Dash(__name__)
# загрузка моделей, датафреймов и скелера 
load_model = load("grid_gbr_model_for_flats.joblib")
scaler = load('scaler_for_dbscan.gz')
scaled_df = pd.read_csv('Scaled_df_for_classifier.csv')
classifier_model = load('Classifier_DBSCAN.joblib')
display_df = pd.read_csv('display_df.csv')


rayon_dict = {'Хорошевский': 59, 'Останкинский': 36, 
              'Кунцево': 17, 'Левобережный': 18, 'Нижегородский': 33, 
              'Митино': 25, 'Покровское-Стрешнево': 40, 'Гольяново': 7, 
              'Обручевский': 35, 'Свиблово': 48, 'Рязанский': 45, 
              'Филевский Парк': 56, 'Басманный': 2, 'Тимирязевский': 55, 
              'Очаково-Матвеевское': 38, 'Богородское': 3, 'Хамовники': 57,
                'Западное Дегунино': 12, 'Лианозово': 20, 'Москворечье-Сабурово': 28,
                  'Хорошево-Мневники': 58, 'Раменки': 43, 'Дорогомилово': 10, 
                  'Замоскворечье': 11, 'Текстильщики': 54, 'Преображенское': 41, 
                  'Сокол': 50, 'Нагатино-Садовники': 29, 'Марфино': 22, 'Отрадное': 37, 
                  'Красносельский': 15, 'Дмитровский': 9, 'Нагатинский Затон': 30, 
                  'Южное Чертаново': 64, 'Савелки': 46, 'Крюково': 16, 'Даниловский': 8,
                    'Северное Измайлово': 49, 'Щукино': 61, 'Царицыно': 60, 'Пресненский': 42, 'Некрасовка': 32, 'Бабушкинский': 1, 'Лефортово': 19, 'Головинский': 6, 'Можайский': 26, 'Якиманка': 66, 'Печатники': 39, 'Метрогородок': 24, 'Солнцево': 52, 'Люблино': 21, 'Южнопортовый': 65, 'Марьина Роща': 23, 'Молжаниновский': 27, 'Войковский': 5, 'Южное Бутово': 62, 'Южное Медведково': 63, 'Ростокино': 44, 'Бутырский': 4, 'Алексеевский': 0, 'Савеловский': 47, 'Ярославский': 67, 'Соколиная Гора': 51, 'Таганский': 53, 'Нагорный': 31, 'Коптево': 13, 'Ново-Переделкино': 34, 'Котловка': 14}
rayon_sorted = ['Алексеевский', 'Бабушкинский', 'Басманный', 'Богородское',
       'Бутырский', 'Войковский', 'Головинский', 'Гольяново',
       'Даниловский', 'Дмитровский', 'Дорогомилово', 'Замоскворечье',
       'Западное Дегунино', 'Коптево', 'Котловка', 'Красносельский',
       'Крюково', 'Кунцево', 'Левобережный', 'Лефортово', 'Лианозово',
       'Люблино', 'Марфино', 'Марьина Роща', 'Метрогородок', 'Митино',
       'Можайский', 'Молжаниновский', 'Москворечье-Сабурово',
       'Нагатино-Садовники', 'Нагатинский Затон', 'Нагорный',
       'Некрасовка', 'Нижегородский', 'Ново-Переделкино', 'Обручевский',
       'Останкинский', 'Отрадное', 'Очаково-Матвеевское', 'Печатники',
       'Покровское-Стрешнево', 'Преображенское', 'Пресненский', 'Раменки',
       'Ростокино', 'Рязанский', 'Савелки', 'Савеловский', 'Свиблово',
       'Северное Измайлово', 'Сокол', 'Соколиная Гора', 'Солнцево',
       'Таганский', 'Текстильщики', 'Тимирязевский', 'Филевский Парк',
       'Хамовники', 'Хорошево-Мневники', 'Хорошевский', 'Царицыно',
       'Щукино', 'Южное Бутово', 'Южное Медведково', 'Южное Чертаново',
       'Южнопортовый', 'Якиманка', 'Ярославский']

clas_dict = {'комфорт':2, 'бизнес':0, 'де-люкс':1,  'премиум':3,  'эконом (панель)':5, 'эконом':4}
clas_list = ['комфорт', 'бизнес', 'де-люкс', 'премиум', 'эконом (панель)', 'эконом']

otdelka_dict = {'Есть': 0, 'Нет': 2, 'Неизвестно': 1}
otdelka_list = ['Есть', 'Нет', 'Неизвестно'] 

zona_dict = {'3) от ~ТТК до МКАД': 2, '4) Москва за МКАД': 3, '2) от СК до ~ТТК' : 1,'1) внутри СК': 0}
zona_sorted = ['1) внутри СК', '2) от СК до ~ТТК', '3) от ~ТТК до МКАД', '4) Москва за МКАД' ]

komnatnost_dict = {'2': 1, '3':2,  '4':3,  'Студия':4, '1':0}
komnatnost_sorted = ['1', '2', '3', '4', 'Студия']

    
def generate_table(dataframe, max_rows=10, labels=0):
    df = dataframe[dataframe['Labels'] == labels]
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ], className="h4 text-bg-dark table table-hover")

cluster_predict = 0

'''tml.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))'''
app.layout = html.Div([
    html.Div([ 

        html.Div([
            html.H4("Зона: ", className="h4 text-bg-dark"),
            dcc.Dropdown(
                    zona_sorted,
                    '1) внутри СК',
                    id='my-zona'
                ),

            html.H4("Район Москвы: ", className="h4 text-bg-dark"),
            dcc.Dropdown(
                    rayon_sorted,
                    'Алексеевский',
                    id='my-rayon'
                ),

            html.H4("Класс: ", className="h4 text-bg-dark"),
            dcc.Dropdown(
                    clas_list,
                    'комфорт',
                    id='my-clas'
                ),

            html.H4("Отделка: ", className="h4 text-bg-dark"),
            dcc.Dropdown(
                    otdelka_list,
                    'Неизвестно',
                    id='my-otdelka'
                ),
            
            html.H4("Площадь: ", className="h4 text-bg-dark"),
            dcc.Input(id = 'my-ploshad', value='0', type='text'),

            html.H4("Комнатность: ", className="h4 text-bg-dark"),
            dcc.Dropdown(
                    komnatnost_sorted,
                    'Студия',
                    id='my-komnatnost'
                ),
            html.Div(html.Button("Узнать цену за квадратный метр", id='calculate', className="btn btn-outline-primary mx-auto"), className="d-flex p-3 text-center"),
            html.H4(id='my-predict', className="h4 text-bg-dark"),
            
            html.H4(children='US Agriculture Exports (2011)'),
            html.Div(id = 'my-table')
        ]), 
    ], className="container gy-5")
], className='min-vh-100 bg-dark')


@app.callback(
    Output(component_id='my-predict', component_property='children'),
    Input(component_id='my-rayon', component_property='value'),
    Input(component_id='my-clas', component_property='value'),
    Input(component_id='my-otdelka', component_property='value'),
    Input(component_id='my-zona', component_property='value'),
    Input(component_id='my-ploshad', component_property='value'),
    Input(component_id='my-komnatnost', component_property='value'),
    Input(component_id='calculate', component_property='n_clicks')
)
def update_output_div(my_rayon,my_clas, my_otdelka, my_zona, my_ploshad, my_komnatnost, n_clicks):
    
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    
    if 'calculate' in changed_id: 

        result = load_model.predict([[rayon_dict[my_rayon], clas_dict[my_clas], otdelka_dict[my_otdelka], zona_dict[my_zona], int(my_ploshad), komnatnost_dict[my_komnatnost]]])
        
        return f'Прогнозируемая цена за кв. м: {result[0]}'
    
@app.callback(
    Output(component_id='my-table', component_property='children'),
    Input(component_id='my-rayon', component_property='value'),
    Input(component_id='my-clas', component_property='value'),
    Input(component_id='my-otdelka', component_property='value'),
    Input(component_id='my-zona', component_property='value'),
    Input(component_id='my-ploshad', component_property='value'),
    Input(component_id='my-komnatnost', component_property='value'),
    Input(component_id='calculate', component_property='n_clicks')
)
def update_output_div(my_rayon,my_clas, my_otdelka, my_zona, my_ploshad, my_komnatnost, n_clicks):
    
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    
    if 'calculate' in changed_id:

        temp = [[rayon_dict[my_rayon], clas_dict[my_clas], otdelka_dict[my_otdelka], zona_dict[my_zona], int(my_ploshad), komnatnost_dict[my_komnatnost]]]

        temp = np.array(temp).reshape(1,-1)
        scaled_data = scaler.transform(temp)
        cluster_predict = classifier_model.predict(scaled_data)

        return generate_table(dataframe= display_df , labels=cluster_predict[0])
        
if __name__ == '__main__':
    app.run_server(debug=True)