"""
Utilitaires de visualisation
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_symptoms_chart(results):
    """Créer un graphique des symptômes"""
    if not results.get('symptoms'):
        fig = go.Figure()
        fig.add_annotation(
            text="No symptoms to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    df = pd.DataFrame(results['symptoms'])
    
    fig = px.bar(
        df,
        x='symptom',
        y='confidence',
        title='Symptom Confidence Scores',
        labels={'symptom': 'Symptom', 'confidence': 'Confidence'},
        color='confidence',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Symptom',
        yaxis_title='Confidence',
        height=400
    )
    
    return fig