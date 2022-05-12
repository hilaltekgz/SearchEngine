from doctest import DocFileCase
from inspect import CO_ITERABLE_COROUTINE
import dash
from dash import no_update
import dash_html_components as html
from dash_html_components.A import A
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_holoniq_components as dhc
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
import spacy

import scipy.spatial
import pickle as pkl
import pandas as pd
nlp = spacy.load("en_core_web_sm")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
def entname(name):
    return html.Span(name, style={
        "font-size": "0.8em",
        "font-weight": "bold",
        "line-height": "1",
        "border-radius": "0.35em",
        "text-transform": "uppercase",
        "vertical-align": "middle",
        "margin-left": "0.5rem"
    })


def entbox(children, color):
    return html.Mark(children, style={
        "background": color,
        "padding": "0.45em 0.6em",
        "margin": "0 0.25em",
        "line-height": "1",
        "border-radius": "0.35em",
    })


def entity(children, name):
    if type(children) is str:
        children = [children]

    children.append(entname(name))
    print("name",name)
    color = '#ff0000'
    return entbox(children, color)



def render(doc):
    children = []
    last_idx = 0
    for ent in doc.ents:
        children.append(doc.text[last_idx:ent.start_char])
        children.append(
            entity(doc.text[ent.start_char:ent.end_char], ent.label_))
        last_idx = ent.end_char
    children.append(doc.text[last_idx:])
    return children



df_sentences = pd.read_csv("/Users/moneye/database/YL/archive/papers.csv")
#df_sentences = df_sentences.set_index("Unnamed: 0")
df = pd.read_csv('/Users/moneye/database/YL/archive/papers.csv', index_col=0)

print(df_sentences[:5])

 

df_sentences = df_sentences["id"].to_dict()
df_sentences_list = list(df_sentences.keys())
df_sentences_list = [str(d) for d in df_sentences_list]
print(len(df_sentences_list))


from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)
#with open(r"C:\Users\Msı\Desktop\Hilal\covid19\covid19\icon_data_bert_search_engine_1\corpus_finetuned_embeddings.pkl" , "rb") as file_:
# corpus_embeddings = pkl.load(file_)
def navbar():
    navbar = dbc.Navbar(
                dbc.Container(
                    [
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(dbc.NavbarBrand("Search Engine", className="ml-2")),
                                ],
                                align="center",
                                no_gutters=True,
                            ),
                        ),
                        dbc.NavbarToggler(id="navbar-toggler2"),
                        dbc.Collapse(
                            dbc.Nav(
                                children= [
                                ],
                                className="ml-auto", navbar=True,
                            ),
                            id="navbar-collapse2",
                            navbar=True,
                        ),
                    ]
                ),
                color="dark",
                dark=True,
                className="mb-5",
            )   
    return navbar




def arama():
    ara =dbc.CardBody(dbc.Row(dbc.Col(html.Div(id="loading-frequencies",className='assets/load.css',
                            children=[
                                dbc.Input(style={'width': '400px',"margin": "15px",'margin-left': '100'},id='mail',debounce=True,placeholder="Search",),
                                dbc.Button('Ara',id='submit-val',n_clicks=0,style={'width': '200px',"margin": "15px"},),

                                dbc.Modal([dbc.ModalHeader("ARTICLE"),
                                           dbc.ModalBody(html.Div(html.Div(id='model_content',className="gradient"),style={"height": "80vh","overflow-y": "scroll"})),]
                                           ,id="modal",
                                           style={"max-width": "none", "width": "90%","height": "80vh"}
                                        ),
                                 ],
                                            )
                        ),
                )
    )
    return ara


def article(text):
        temp = []
        card = {}
        paper_id = []



        word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        query_embeddings = model.encode([text],show_progress_bar=True)

        closest_n = 5
        for query, query_embedding in zip(text, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
            print("distances",distances)
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            count = 1
            for idx, distance in results[0:closest_n]:
                print('count',count)
                row_dict = df.loc[df.index== corpus[idx]].to_dict()
                doc = nlp(df["abstract"].loc[idx])
                print("graph-page"+str(count)+"")
                print(idx)
                card[idx] = dbc.Container(dbc.NavLink(html.Div(id="graph-page"+str(count)+"",children=[html.H3(df["title"].loc[idx], className="card-title",style={'color':'black'}),
                                                html.P(render(doc),className="card-text",style={'line-height': '2.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},),
                                                html.P('Score: '+str(round((1-distance),4))+'',className="card-text",style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','font-weight': 'bold','text-align': 'justify','text-justify': 'inter-word'},),
                                                html.P(dbc.Button(df["title"].loc[idx] , id="buton_modal"+str(count)+"", n_clicks=0,color="secondary",outline=True,),className="card-text",style={'color': 'black'},),],style={'line-height': '1.5','padding':'0.5%','display': 'none','marginLeft': 'auto', 'marginRight': 'auto'} ),target='blank_' ))
                count = count + 1
        graph = [ dbc.Container(dbc.Row([value for value in card.values()]),)]


        app.layout = html.Div([home_page(),dbc.Row(dbc.Col(graph))])

        return graph



def article1(text):
        temp = []
        card = {}
        paper_id = []

        word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')

        
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        query_embeddings = model.encode([text],show_progress_bar=True)

        
        closest_n = 5
        for query, query_embedding in zip(text, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            count = 1
            for idx, distance in results[0:closest_n]:
                print('count',count)
                row_dict = df.loc[df.index== corpus[idx]].to_dict()
                doc = nlp(df["abstract"].loc[idx])
                print("graph-page"+str(count)+"")
                print(idx)
                card[idx] = dbc.Container(dbc.NavLink( html.Div(id="graph-page"+str(count)+"",children=[html.H3(df["title"].loc[idx], className="card-title",style={'color':'black'}),
                                                html.P(render(doc),className="card-text",style={'line-height': '2.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},),
                                                html.P('Score: '+str(round((1-distance),4))+'',className="card-text",style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','font-weight': 'bold','text-align': 'justify','text-justify': 'inter-word'},),
                                                html.P(dbc.Button(df["title"].loc[idx] , id="buton_modal"+str(count)+"", n_clicks=0,color="secondary",outline=True,),className="card-text",style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},),],style={'padding':'0.5%','display': 'block'} ),target='blank_' ))
                count = count + 1


        graph = [dbc.Container(dbc.Row([value for value in card.values()]),)]


        app.layout = html.Div([home_page(),dbc.Row(dbc.Col(graph))])

        return graph






def home_page():
    layout = html.Div(style={'background-position': 'center'},
     children = [
            navbar(),
            dbc.Container(arama()),
        ]                   
        ) #end div
    return layout


app.layout = html.Div([
            home_page(),
            dbc.Row(
                dbc.Col(article("hello")))
        ])

from dash.exceptions import PreventUpdate


@app.callback(
    [dash.dependencies.Output("graph-page1","style"),
    dash.dependencies.Output("graph-page2","style"),
    dash.dependencies.Output("graph-page3","style"),
    dash.dependencies.Output("graph-page4","style"),
    dash.dependencies.Output("graph-page5","style")],     
    [dash.dependencies.Input("mail","value"),
    dash.dependencies.Input("submit-val","n_clicks")]
    )
def dsply(firma,value):
    
    ctx = dash.callback_context

    if (value == None or firma == None or firma == []):
        return no_update,no_update,no_update,no_update,no_update
    else:
        if (ctx.triggered and ctx.triggered[0]['value'] != 0):
            if ctx.triggered[0]['prop_id'].split('.')[0]=='submit-val':
                return {'display': 'block'},{'display': 'block'},{'display': 'block'},{'display': 'block'},{'display': 'block'}
            raise PreventUpdate
        raise PreventUpdate



@app.callback(
    [dash.dependencies.Output("graph-page1","children"),
    dash.dependencies.Output("graph-page2","children"),
    dash.dependencies.Output("graph-page3","children"),
    dash.dependencies.Output("graph-page4","children"),
    dash.dependencies.Output("graph-page5","children"),
    dash.dependencies.Output("model_content","children")],     
    [dash.dependencies.Input("mail","value"),
    dash.dependencies.Input("submit-val","n_clicks"),
    dash.dependencies.Input("buton_modal1","n_clicks"),
    dash.dependencies.Input("buton_modal2","n_clicks"),
    dash.dependencies.Input("buton_modal3","n_clicks"),
    dash.dependencies.Input("buton_modal4","n_clicks"),
    dash.dependencies.Input("buton_modal5","n_clicks")],
    prevent_initial_call=True)
def graphh(firma,value,n1,n2,n3,n4,n5):
    if (value == None or firma == None or firma == []):
        return no_update,no_update,no_update,no_update,no_update,no_update

    else:
        temp = []
        card = {}
        popup = {}
        paper_id = []



        word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')

        
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        query_embeddings = model.encode([firma],show_progress_bar=True)

        
        closest_n = 5
        for query, query_embedding in zip(firma, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            count = 1
            for idx, distance in results[0:closest_n]:
                print('count',count)
                row_dict = df.loc[df.index== corpus[idx]].to_dict()
                print('row_dict',row_dict)
                doc = nlp(str(df["abstract"].loc[idx]))
                print("graph-page"+str(count)+"")
                print(idx)
                card[idx] = dbc.Container(dbc.NavLink( html.Div(id="graph-page"+str(count)+"",children=[html.H3(df["title"].loc[idx], className="card-title",style={'color':'black'}),
                                                html.P(
                                                            render(doc),
                                                            #row_dict["abstract_summary"][corpus[idx]].replace('<br>', ' '),
                                                            className="card-text",
                                                            style={'line-height': '2.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                                                html.P(
                                                            'Score: '+str(round((1-distance),4))+'',
                                                            className="card-text",
                                                            style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','font-weight': 'bold','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                                                html.P(
                                                            dbc.Button(df["title"].loc[idx] , id="buton_modal"+str(count)+"", n_clicks=0,color="secondary",outline=True,),
                                                            className="card-text",
                                                            style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                            ],style={'padding':'0.5%','display': 'block'} ),target='blank_' ))

                popup[idx] = dbc.Container(dbc.NavLink( html.Div(id="graph-page8"+str(count)+"",children=[html.H4(df["title"].loc[idx], className="card-title",style={'color':'black'}),
                                                html.H6("Introduction",style={'color':'black'}),
                                                html.P(     df["paper_text"].loc[idx],
                                                            className="card-text",
                                                            style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                                                html.H6("Method",style={'color':'black'}),
                                                html.P(     df["paper_text"].loc[idx],
                                                            className="card-text",
                                                            style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                                                html.H6("Result",style={'color':'black'}),
                                                html.P(     df["paper_text"].loc[idx],
                                                            className="card-text",  
                                                            style={'line-height': '1.5','color': 'black','marginLeft': 'auto', 'marginRight': 'auto','text-align': 'justify','text-justify': 'inter-word'},
                                                ),
                            ],style={'padding':'0.5%','display': 'block'}),target='blank_' ))

                count = count + 1


        ozet = []
        for value in card.values():
            ozet.append(value)

        makale = []
        for mak in popup.values():
            makale.append(mak)


        graph = [dbc.Container(dbc.Row([value for value in card.values()]))]
        graph1 = [ dbc.Container(dbc.Row([ozet[0]]))]
        graph2 = [ dbc.Container(dbc.Row([ozet[1]]))]
        graph3 = [ dbc.Container(dbc.Row([ozet[2]]))]
        graph4 = [ dbc.Container(dbc.Row([ozet[3]]))]
        graph5 = [ dbc.Container(dbc.Row([ozet[4]]))]

        #model1 = [ dbc.Container(dbc.Row([makale[0]]))]
        #model2 = [ dbc.Container(dbc.Row([makale[1]]))]
        #model3 = [ dbc.Container(dbc.Row([makale[2]]))]
        #model4 = [ dbc.Container(dbc.Row([makale[3]]))]
        #model5 = [ dbc.Container(dbc.Row([makale[4]]))]

        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

       
        if button_id=='buton_modal1':
            model = [dbc.Container(dbc.Row([makale[0]]))]
        elif button_id=='buton_modal2':
            model = [dbc.Container(dbc.Row([makale[1]]))]
        elif button_id=='buton_modal3':
            model = [dbc.Container(dbc.Row([makale[2]]))]
        elif button_id=='buton_modal4':
            model = [dbc.Container(dbc.Row([makale[3]]))]
        elif button_id=='buton_modal5':
            model = [dbc.Container(dbc.Row([makale[4]]))]
        else:
            model = ""


        app.layout = html.Div([
            home_page(),
            dbc.Row(
                dbc.Col(graph))])

        return graph1,graph2,graph3,graph4,graph5,model



@app.callback(
    dash.dependencies.Output("modal","is_open"),
    [dash.dependencies.Input("buton_modal1","n_clicks"),
    dash.dependencies.Input("buton_modal2","n_clicks"),
    dash.dependencies.Input("buton_modal3","n_clicks"),     
    dash.dependencies.Input("buton_modal4","n_clicks"),
    dash.dependencies.Input("buton_modal5","n_clicks")],
    [dash.dependencies.State("modal","is_open")]
    )
def graphh(n1,n2,n3,n4,n5, is_open):
    if n1 or n2 or n3 or n4 or n5:
        return not is_open
    return  is_open



if __name__ == '__main__':

    
    app.run_server(port=8050,debug=True) 
