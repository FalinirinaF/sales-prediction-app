import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Ventes Mensuelles",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

class SalesPredictionApp:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.model_results = {}
        
    def run(self):
        st.markdown('<h1 class="main-header">📊 Application de Prédiction des Ventes Mensuelles</h1>', unsafe_allow_html=True)
        
        # Sidebar pour la navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choisissez une section:",
            ["🏠 Accueil", "📁 Import des Données", "🔧 Préprocessing", 
             "🤖 Modélisation", "📊 Visualisations", "🔮 Prédictions"]
        )
        
        if page == "🏠 Accueil":
            self.show_home()
        elif page == "📁 Import des Données":
            self.show_data_import()
        elif page == "🔧 Préprocessing":
            self.show_preprocessing()
        elif page == "🤖 Modélisation":
            self.show_modeling()
        elif page == "📊 Visualisations":
            self.show_visualizations()
        elif page == "🔮 Prédictions":
            self.show_predictions()

    def show_home(self):
        st.markdown("""
        ## Bienvenue dans l'Application de Prédiction des Ventes Mensuelles
        
        Cette application vous permet d'analyser vos données de ventes historiques et de prédire les ventes futures.
        
        ### Fonctionnalités principales:
        - **Import de données**: Chargez vos fichiers CSV ou Excel
        - **Préprocessing**: Nettoyez et préparez vos données
        - **Modélisation**: Entraînez des modèles de machine learning
        - **Visualisations**: Explorez vos données avec des graphiques interactifs
        - **Prédictions**: Générez des prédictions pour les mois futurs
        
        ### Pour commencer:
        1. Allez dans la section "Import des Données" pour charger votre fichier
        2. Utilisez le "Préprocessing" pour nettoyer vos données
        3. Entraînez vos modèles dans la section "Modélisation"
        4. Explorez vos résultats dans "Visualisations" et "Prédictions"
        """)

    def show_data_import(self):
        st.header("📁 Import des Données")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV ou Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Formats supportés: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier
                if uploaded_file.name.endswith('.csv'):
                    self.data = pd.read_csv(uploaded_file)
                else:
                    self.data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Fichier chargé avec succès! {len(self.data)} lignes importées.")
                
                # Aperçu des données
                st.subheader("Aperçu des données")
                st.dataframe(self.data.head())
                
                # Informations sur le dataset
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de lignes", len(self.data))
                with col2:
                    st.metric("Nombre de colonnes", len(self.data.columns))
                with col3:
                    st.metric("Taille mémoire", f"{self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                # Informations sur les colonnes
                st.subheader("Informations sur les colonnes")
                col_info = pd.DataFrame({
                    'Type': self.data.dtypes,
                    'Valeurs manquantes': self.data.isnull().sum(),
                    'Valeurs uniques': self.data.nunique()
                })
                st.dataframe(col_info)
                
                # Sauvegarde dans session state
                st.session_state['data'] = self.data
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        
        # Récupération des données depuis session state
        if 'data' in st.session_state:
            self.data = st.session_state['data']

    def show_preprocessing(self):
        st.header("🔧 Préprocessing des Données")
        
        if self.data is None and 'data' in st.session_state:
            self.data = st.session_state['data']
        
        if self.data is None:
            st.warning("⚠️ Veuillez d'abord importer des données dans la section 'Import des Données'.")
            return
        
        st.subheader("Configuration du préprocessing")
        
        # Vérification et mapping des colonnes requises
        required_columns = {
            'date_block_num': 'Numéro du mois (date_block_num)',
            'shop_id': 'ID du magasin (shop_id)', 
            'item_id': 'ID de l\'article (item_id)',
            'item_cnt_day': 'Ventes quotidiennes (item_cnt_day)'
        }
        
        # Vérifier quelles colonnes existent
        available_columns = list(self.data.columns)
        missing_columns = []
        column_mapping = {}
        
        st.subheader("Mapping des colonnes")
        st.write("Associez vos colonnes aux champs requis:")
        
        for req_col, description in required_columns.items():
            if req_col in available_columns:
                column_mapping[req_col] = req_col
                st.success(f"✅ {description}: trouvé automatiquement")
            else:
                selected_col = st.selectbox(
                    f"Sélectionnez la colonne pour {description}:",
                    [''] + available_columns,
                    key=f"mapping_{req_col}"
                )
                if selected_col:
                    column_mapping[req_col] = selected_col
                else:
                    missing_columns.append(req_col)
        
        if missing_columns:
            st.error(f"❌ Colonnes manquantes: {', '.join(missing_columns)}")
            st.stop()
        
        # Renommer les colonnes selon le mapping
        data_mapped = self.data.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Options de préprocessing
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Options de nettoyage")
            remove_outliers = st.checkbox("Supprimer les valeurs aberrantes", value=True)
            outlier_method = st.selectbox("Méthode de détection:", ["IQR", "Z-score"])
            fill_missing = st.checkbox("Remplir les valeurs manquantes", value=True)
            
        with col2:
            st.subheader("Agrégation")
            aggregate_monthly = st.checkbox("Agréger par mois", value=True)
            min_sales_threshold = st.number_input("Seuil minimum de ventes:", min_value=0, value=0)
        
        if st.button("🔄 Appliquer le préprocessing"):
            try:
                processed_data = data_mapped.copy()
                
                # Vérifier que les colonnes nécessaires existent
                sales_col = 'item_cnt_day'
                if sales_col not in processed_data.columns:
                    st.error(f"❌ Colonne '{sales_col}' non trouvée après mapping")
                    return
                
                # Statistiques avant traitement
                st.subheader("📊 Statistiques avant traitement")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lignes totales", len(processed_data))
                with col2:
                    st.metric("Valeurs manquantes", processed_data.isnull().sum().sum())
                with col3:
                    if sales_col in processed_data.columns:
                        st.metric("Ventes moyennes", f"{processed_data[sales_col].mean():.2f}")
                
                # Nettoyage des valeurs manquantes
                if fill_missing:
                    if sales_col in processed_data.columns:
                        processed_data[sales_col] = processed_data[sales_col].fillna(0)
                    processed_data = processed_data.fillna(method='ffill').fillna(0)
                
                # Suppression des valeurs aberrantes
                if remove_outliers and sales_col in processed_data.columns:
                    if outlier_method == "IQR":
                        Q1 = processed_data[sales_col].quantile(0.25)
                        Q3 = processed_data[sales_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        processed_data = processed_data[
                            (processed_data[sales_col] >= lower_bound) & 
                            (processed_data[sales_col] <= upper_bound)
                        ]
                    else:  # Z-score
                        z_scores = np.abs((processed_data[sales_col] - processed_data[sales_col].mean()) / processed_data[sales_col].std())
                        processed_data = processed_data[z_scores < 3]
                
                # Agrégation mensuelle
                if aggregate_monthly:
                    # Vérifier que les colonnes nécessaires existent
                    group_cols = []
                    if 'date_block_num' in processed_data.columns:
                        group_cols.append('date_block_num')
                    if 'shop_id' in processed_data.columns:
                        group_cols.append('shop_id')
                    if 'item_id' in processed_data.columns:
                        group_cols.append('item_id')
                    
                    if group_cols and sales_col in processed_data.columns:
                        processed_data = processed_data.groupby(group_cols)[sales_col].sum().reset_index()
                        processed_data.columns = group_cols + ['item_cnt_month']
                        sales_col = 'item_cnt_month'
                
                # Filtrage par seuil de ventes
                if min_sales_threshold > 0 and sales_col in processed_data.columns:
                    processed_data = processed_data[processed_data[sales_col] >= min_sales_threshold]
                
                # Statistiques après traitement
                st.subheader("📊 Statistiques après traitement")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lignes restantes", len(processed_data))
                with col2:
                    st.metric("Valeurs manquantes", processed_data.isnull().sum().sum())
                with col3:
                    if sales_col in processed_data.columns:
                        st.metric("Ventes moyennes", f"{processed_data[sales_col].mean():.2f}")
                
                # Aperçu des données traitées
                st.subheader("Aperçu des données traitées")
                st.dataframe(processed_data.head())
                
                # Graphiques de comparaison
                if sales_col in processed_data.columns and sales_col in data_mapped.columns:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Avant traitement', 'Après traitement'))
                    
                    fig.add_trace(
                        go.Histogram(x=data_mapped[sales_col], name='Avant', nbinsx=50),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Histogram(x=processed_data[sales_col], name='Après', nbinsx=50),
                        row=1, col=2
                    )
                    
                    fig.update_layout(title="Distribution des ventes avant/après traitement")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarde
                self.processed_data = processed_data
                st.session_state['processed_data'] = processed_data
                
                st.success("✅ Préprocessing terminé avec succès!")
                
                # Option de téléchargement
                csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les données traitées",
                    data=csv,
                    file_name="donnees_traitees.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors du préprocessing: {str(e)}")

    def show_modeling(self):
        st.header("🤖 Modélisation et Entraînement")
        
        if 'processed_data' in st.session_state:
            self.processed_data = st.session_state['processed_data']
        
        if self.processed_data is None:
            st.warning("⚠️ Veuillez d'abord traiter vos données dans la section 'Préprocessing'.")
            return
        
        st.subheader("Configuration de l'entraînement")
        
        # Sélection de la colonne cible
        target_columns = [col for col in self.processed_data.columns if 'cnt' in col.lower() or 'sales' in col.lower()]
        if not target_columns:
            target_columns = list(self.processed_data.select_dtypes(include=[np.number]).columns)
        
        target_col = st.selectbox("Sélectionnez la colonne cible (ventes):", target_columns)
        
        # Configuration des features
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Features à utiliser")
            use_lag_features = st.checkbox("Features de lag (1, 2, 3 mois)", value=True)
            use_shop_features = st.checkbox("Features agrégées par magasin", value=True)
            use_category_features = st.checkbox("Features agrégées par catégorie", value=False)
            
        with col2:
            st.subheader("Paramètres d'entraînement")
            test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random state", value=42)
        
        if st.button("🚀 Entraîner les modèles"):
            try:
                with st.spinner("Entraînement en cours..."):
                    # Préparation des données
                    model_data = self.processed_data.copy()
                    sales_col = target_col
                    
                    # Features de lag si demandées
                    if use_lag_features:
                        model_data = model_data.reset_index(drop=True)
                        
                        for lag in [1, 2, 3]:
                            lag_series = model_data.groupby(['shop_id', 'item_id'])[sales_col].shift(lag)
                            model_data[f'{sales_col}_lag_{lag}'] = lag_series.values
                        
                        rolling_mean = model_data.groupby(['shop_id', 'item_id'])[sales_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
                        model_data[f'{sales_col}_ma_3'] = rolling_mean
                    
                    # Features agrégées par magasin
                    if use_shop_features:
                        shop_stats = model_data.groupby(['shop_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        shop_stats.columns = ['shop_id', 'date_block_num', 'shop_avg_sales', 'shop_std_sales']
                        shop_stats = shop_stats.fillna(0)
                        model_data = model_data.merge(shop_stats, on=['shop_id', 'date_block_num'], how='left')
                    
                    # Features agrégées par catégorie
                    if use_category_features:
                        category_stats = model_data.groupby(['item_category_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        category_stats.columns = ['item_category_id', 'date_block_num', 'category_avg_sales', 'category_std_sales']
                        category_stats = category_stats.fillna(0)
                        model_data = model_data.merge(category_stats, on=['item_category_id', 'date_block_num'], how='left')
                    
                    # Suppression des NaN
                    model_data = model_data.fillna(0)
                    
                    # Sélection des features
                    feature_cols = [col for col in model_data.columns if col != sales_col and col not in ['item_id', 'shop_id']]
                    X = model_data[feature_cols]
                    y = model_data[sales_col]
                    
                    # Division train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Entraînement des modèles
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state)
                    }
                    
                    results = {}
                    
                    for name, model in models.items():
                        # Entraînement
                        model.fit(X_train, y_train)
                        
                        # Prédictions
                        y_pred = model.predict(X_test)
                        
                        # Métriques
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[name] = {
                            'model': model,
                            'predictions': y_pred,
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2
                        }
                    
                    # Sauvegarde des résultats
                    self.models = {name: result['model'] for name, result in results.items()}
                    self.model_results = results
                    st.session_state['models'] = self.models
                    st.session_state['model_results'] = results
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                st.success("✅ Entraînement terminé!")
                
                # Affichage des résultats
                st.subheader("📊 Résultats des modèles")
                
                # Tableau de comparaison
                comparison_df = pd.DataFrame({
                    'Modèle': list(results.keys()),
                    'MAE': [results[name]['mae'] for name in results.keys()],
                    'RMSE': [results[name]['rmse'] for name in results.keys()],
                    'R²': [results[name]['r2'] for name in results.keys()]
                })
                
                st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'RMSE']).highlight_max(subset=['R²']))
                
                # Graphiques de performance
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('MAE', 'RMSE', 'R²', 'Prédictions vs Réalité'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # MAE
                fig.add_trace(
                    go.Bar(x=list(results.keys()), y=[results[name]['mae'] for name in results.keys()], name='MAE'),
                    row=1, col=1
                )
                
                # RMSE
                fig.add_trace(
                    go.Bar(x=list(results.keys()), y=[results[name]['rmse'] for name in results.keys()], name='RMSE'),
                    row=1, col=2
                )
                
                # R²
                fig.add_trace(
                    go.Bar(x=list(results.keys()), y=[results[name]['r2'] for name in results.keys()], name='R²'),
                    row=2, col=1
                )
                
                # Prédictions vs Réalité (meilleur modèle)
                best_model = min(results.keys(), key=lambda x: results[x]['mae'])
                fig.add_trace(
                    go.Scatter(x=y_test, y=results[best_model]['predictions'], mode='markers', name=f'{best_model}'),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                              mode='lines', name='Ligne parfaite', line=dict(dash='dash')),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, title="Performance des modèles")
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance des features (pour Random Forest)
                if 'Random Forest' in results:
                    st.subheader("🎯 Importance des features (Random Forest)")
                    rf_model = results['Random Forest']['model']
                    feature_importance = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h')
                    fig.update_layout(title="Top 10 des features les plus importantes")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommandations
                st.subheader("💡 Recommandations")
                best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
                st.info(f"""
                **Meilleur modèle**: {best_model_name}
                - MAE: {results[best_model_name]['mae']:.2f}
                - RMSE: {results[best_model_name]['rmse']:.2f}
                - R²: {results[best_model_name]['r2']:.3f}
                
                **Conseils d'amélioration**:
                - Si R² < 0.7: Considérez ajouter plus de features ou de données
                - Si MAE élevé: Vérifiez la qualité des données et les outliers
                - Testez différents hyperparamètres pour optimiser les performances
                """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")

    def show_visualizations(self):
        st.header("📊 Visualisations")
        
        if 'processed_data' in st.session_state:
            self.processed_data = st.session_state['processed_data']
        
        if self.processed_data is None:
            st.warning("⚠️ Veuillez d'abord traiter vos données dans la section 'Préprocessing'.")
            return
        
        st.subheader("Graphiques des ventes")
        
        # Graphique de ventes totales par mois
        fig = px.line(self.processed_data, x='date_block_num', y='item_cnt_day', title='Ventes totales par mois')
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de ventes moyennes par magasin
        shop_avg_sales = self.processed_data.groupby('shop_id')['item_cnt_day'].mean().reset_index()
        fig = px.bar(shop_avg_sales, x='shop_id', y='item_cnt_day', title='Ventes moyennes par magasin')
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de ventes moyennes par catégorie
        category_avg_sales = self.processed_data.groupby('item_category_id')['item_cnt_day'].mean().reset_index()
        fig = px.bar(category_avg_sales, x='item_category_id', y='item_cnt_day', title='Ventes moyennes par catégorie')
        st.plotly_chart(fig, use_container_width=True)

    def show_predictions(self):
        st.header("🔮 Prédictions")
        
        if 'models' in st.session_state and 'X_test' in st.session_state and 'y_test' in st.session_state:
            self.models = st.session_state['models']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
        
        if not self.models or X_test is None or y_test is None:
            st.warning("⚠️ Veuillez d'abord entraîner vos modèles dans la section 'Modélisation'.")
            return
        
        st.subheader("Sélectionnez un modèle pour les prédictions")
        model_name = st.selectbox("Modèle:", list(self.models.keys()))
        
        if st.button("🔮 Générer les prédictions"):
            try:
                model = self.models[model_name]
                predictions = model.predict(X_test)
                
                st.subheader("Prédictions vs Réalité")
                fig = px.scatter(x=y_test, y=predictions, title=f"Prédictions vs Réalité ({model_name})")
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                          mode='lines', name='Ligne parfaite', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
                # Téléchargement des prédictions
                predictions_df = pd.DataFrame({
                    'Ventes réelles': y_test,
                    'Prédictions': predictions
                })
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les prédictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération des prédictions : {str(e)}")

if __name__ == "__main__":
    app = SalesPredictionApp()
    app.run()
