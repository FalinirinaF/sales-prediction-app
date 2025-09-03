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
import io
import base64
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction des Ventes Mensuelles",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
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
        self.predictions = None
        
    def run(self):
        """Point d'entrée principal de l'application"""
        st.markdown('<h1 class="main-header">📊 Prédiction des Ventes Mensuelles</h1>', unsafe_allow_html=True)
        
        # Sidebar pour la navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choisir une section",
            ["🏠 Accueil", "📁 Import des Données", "🔧 Prétraitement", "🤖 Modélisation", "📈 Visualisations", "📊 Prédictions"]
        )
        
        # Routage des pages
        if page == "🏠 Accueil":
            self.show_home()
        elif page == "📁 Import des Données":
            self.show_data_import()
        elif page == "🔧 Prétraitement":
            self.show_preprocessing()
        elif page == "🤖 Modélisation":
            self.show_modeling()
        elif page == "📈 Visualisations":
            self.show_visualizations()
        elif page == "📊 Prédictions":
            self.show_predictions()
    
    def show_home(self):
        """Page d'accueil avec présentation de l'application"""
        st.markdown("""
        ## Bienvenue dans l'application de prédiction des ventes mensuelles
        
        Cette application vous permet de :
        
        ### 📁 **Import des Données**
        - Charger vos données de ventes historiques (CSV/Excel)
        - Visualiser un aperçu de vos données
        
        ### 🔧 **Prétraitement**
        - Nettoyer automatiquement les données
        - Agréger les ventes quotidiennes en ventes mensuelles
        - Transformer les données pour l'analyse
        
        ### 🤖 **Modélisation**
        - Entraîner des modèles de machine learning
        - Comparer les performances (Linear Regression, Random Forest, XGBoost)
        
        ### 📈 **Visualisations**
        - Analyser les tendances historiques
        - Visualiser les prédictions futures
        
        ### 📊 **Prédictions**
        - Faire des prédictions personnalisées
        - Exporter les résultats en CSV
        
        ---
        
        ### Format des données attendu :
        
        | Colonne | Description |
        |---------|-------------|
        | ID | Identifiant unique (Shop, Item) |
        | shop_id | Identifiant du magasin |
        | item_id | Identifiant du produit |
        | item_category_id | Identifiant de la catégorie |
        | item_cnt_day | Nombre de produits vendus par jour |
        | item_price | Prix de l'article |
        | date | Date de la vente (dd/mm/yyyy) |
        | date_block_num | Numéro du mois (0 = janv 2013) |
        | item_name | Nom du produit |
        | shop_name | Nom du magasin |
        | item_category_name | Nom de la catégorie |
        
        **Commencez par importer vos données dans la section "Import des Données" !**
        """)
        
        # Affichage des statistiques si des données sont chargées
        if 'sales_data' in st.session_state:
            st.success("✅ Données chargées avec succès !")
            data = st.session_state['sales_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nombre de lignes", f"{len(data):,}")
            with col2:
                st.metric("Nombre de magasins", data['shop_id'].nunique())
            with col3:
                st.metric("Nombre de produits", data['item_id'].nunique())
            with col4:
                st.metric("Période", f"{data['date_block_num'].min()} - {data['date_block_num'].max()}")
    
    def show_data_import(self):
        """Interface d'import des données"""
        st.header("📁 Import des Données")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV ou Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Uploadez votre fichier contenant les données de ventes historiques"
        )
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier selon son type
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès !")
                
                # Sauvegarde dans session state
                st.session_state['sales_data'] = data
                self.data = data
                
                # Affichage des informations sur le dataset
                st.subheader("📊 Aperçu des données")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre de lignes", len(data))
                    st.metric("Nombre de colonnes", len(data.columns))
                
                with col2:
                    st.metric("Taille mémoire", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    st.metric("Valeurs manquantes", data.isnull().sum().sum())
                
                # Aperçu des premières lignes
                st.subheader("🔍 Premières lignes")
                st.dataframe(data.head(10))
                
                # Informations sur les colonnes
                st.subheader("📋 Informations sur les colonnes")
                col_info = pd.DataFrame({
                    'Type': data.dtypes,
                    'Valeurs non-nulles': data.count(),
                    'Valeurs manquantes': data.isnull().sum(),
                    'Valeurs uniques': data.nunique()
                })
                st.dataframe(col_info)
                
                # Vérification des colonnes requises
                required_columns = [
                    'shop_id', 'item_id', 'item_category_id', 'item_cnt_day',
                    'item_price', 'date', 'date_block_num', 'item_name',
                    'shop_name', 'item_category_name'
                ]
                
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"❌ Colonnes manquantes : {', '.join(missing_columns)}")
                else:
                    st.success("✅ Toutes les colonnes requises sont présentes !")
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")
        
        else:
            st.info("👆 Veuillez uploader un fichier pour commencer l'analyse")
    
    def show_preprocessing(self):
        """Interface de prétraitement des données"""
        st.header("🔧 Prétraitement des Données")
        
        if 'sales_data' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données dans la section 'Import des Données'")
            return
        
        data = st.session_state['sales_data']
        
        st.subheader("📊 État actuel des données")
        
        # Affichage des statistiques avant preprocessing
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lignes totales", f"{len(data):,}")
        with col2:
            st.metric("Valeurs manquantes", f"{data.isnull().sum().sum():,}")
        with col3:
            st.metric("Doublons", f"{data.duplicated().sum():,}")
        with col4:
            st.metric("Période (mois)", f"{data['date_block_num'].nunique()}")
        
        # Options de prétraitement
        st.subheader("⚙️ Options de prétraitement")
        
        col1, col2 = st.columns(2)
        with col1:
            remove_duplicates = st.checkbox("Supprimer les doublons", value=True)
            handle_missing = st.checkbox("Traiter les valeurs manquantes", value=True)
            remove_outliers = st.checkbox("Supprimer les valeurs aberrantes", value=False)
        
        with col2:
            aggregate_monthly = st.checkbox("Agréger en ventes mensuelles", value=True)
            min_sales_threshold = st.number_input("Seuil minimum de ventes", min_value=0, value=0)
            date_range = st.slider(
                "Plage de mois à conserver",
                min_value=int(data['date_block_num'].min()),
                max_value=int(data['date_block_num'].max()),
                value=(int(data['date_block_num'].min()), int(data['date_block_num'].max()))
            )
        # Bouton de prétraitement
if st.button("🚀 Lancer le prétraitement", type="primary"):
    with st.spinner("Prétraitement en cours..."):
        processed_data = data.copy()

        # Filtrage par plage de dates
        processed_data = processed_data[
            (processed_data['date_block_num'] >= date_range[0]) & 
            (processed_data['date_block_num'] <= date_range[1])
        ]

        # Suppression des doublons
        if remove_duplicates:
            initial_rows = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            st.info(f"✅ {initial_rows - len(processed_data):,} doublons supprimés")

        # Traitement des valeurs manquantes
        if handle_missing:
            # Vérification et remplissage item_price
            if 'item_price' in processed_data.columns:
                if processed_data['item_price'].isnull().any():
                    processed_data['item_price'] = processed_data.groupby('item_category_id')['item_price'].transform(
                        lambda x: x.fillna(x.median())
                    )
            else:
                processed_data['item_price'] = 0  # valeur par défaut si colonne absente

            # Vérification et remplissage item_cnt_day
            if 'item_cnt_day' in processed_data.columns:
                processed_data['item_cnt_day'] = processed_data['item_cnt_day'].fillna(0)
            else:
                processed_data['item_cnt_day'] = 0

            # Colonnes textuelles
            text_columns = ['item_name', 'shop_name', 'item_category_name']
            for col in text_columns:
                if col not in processed_data.columns:
                    processed_data[col] = 'Inconnu'
                else:
                    processed_data[col] = processed_data[col].fillna('Inconnu')

            st.info("✅ Valeurs manquantes traitées")

        # Suppression des valeurs aberrantes
        if remove_outliers:
            if 'item_price' in processed_data.columns:
                q99 = processed_data['item_price'].quantile(0.99)
                initial_rows = len(processed_data)
                processed_data = processed_data[
                    (processed_data['item_price'] > 0) & 
                    (processed_data['item_price'] <= q99)
                ]
            if 'item_cnt_day' in processed_data.columns:
                q99_qty = processed_data['item_cnt_day'].quantile(0.99)
                processed_data = processed_data[
                    (processed_data['item_cnt_day'] >= 0) & 
                    (processed_data['item_cnt_day'] <= q99_qty)
                ]
            st.info(f"✅ Valeurs aberrantes supprimées")

        # Agrégation mensuelle
        if aggregate_monthly:
            # S'assurer que toutes les colonnes nécessaires existent
            for col in ['shop_id', 'item_id', 'date_block_num', 'item_cnt_day', 'item_price', 
                        'item_category_id', 'item_name', 'shop_name', 'item_category_name']:
                if col not in processed_data.columns:
                    processed_data[col] = 0 if col in ['item_cnt_day', 'item_price', 'item_category_id'] else 'Inconnu'

            monthly_data = processed_data.groupby(['shop_id', 'item_id', 'date_block_num']).agg({
                'item_cnt_day': 'sum',
                'item_price': 'mean',
                'item_category_id': 'first',
                'item_name': 'first',
                'shop_name': 'first',
                'item_category_name': 'first'
            }).reset_index()

            monthly_data = monthly_data.rename(columns={'item_cnt_day': 'item_cnt_month'})

            # Filtrage par seuil de ventes
            if min_sales_threshold > 0:
                initial_rows = len(monthly_data)
                monthly_data = monthly_data[monthly_data['item_cnt_month'] >= min_sales_threshold]
                st.info(f"✅ {initial_rows - len(monthly_data):,} lignes avec ventes < {min_sales_threshold} supprimées")

            processed_data = monthly_data
            st.info("✅ Données agrégées en ventes mensuelles")

        # Sauvegarde des données prétraitées
        st.session_state['processed_data'] = processed_data
        st.success("🎉 Prétraitement terminé avec succès !")

        # # Bouton de prétraitement
        # if st.button("🚀 Lancer le prétraitement", type="primary"):
        #     with st.spinner("Prétraitement en cours..."):
        #         processed_data = data.copy()
                
        #         # Filtrage par plage de dates
        #         processed_data = processed_data[
        #             (processed_data['date_block_num'] >= date_range[0]) & 
        #             (processed_data['date_block_num'] <= date_range[1])
        #         ]
                
        #         # Suppression des doublons
        #         if remove_duplicates:
        #             initial_rows = len(processed_data)
        #             processed_data = processed_data.drop_duplicates()
        #             st.info(f"✅ {initial_rows - len(processed_data):,} doublons supprimés")
                
        #         # Traitement des valeurs manquantes
        #         if handle_missing:
        #             # Prix manquants : médiane par catégorie
        #             if processed_data['item_price'].isnull().any():
        #                 processed_data['item_price'] = processed_data.groupby('item_category_id')['item_price'].transform(
        #                     lambda x: x.fillna(x.median())
        #                 )
                    
        #             # Quantités manquantes : 0
        #             processed_data['item_cnt_day'] = processed_data['item_cnt_day'].fillna(0)
                    
        #             # Autres colonnes textuelles
        #             text_columns = ['item_name', 'shop_name', 'item_category_name']
        #             for col in text_columns:
        #                 if col in processed_data.columns:
        #                     processed_data[col] = processed_data[col].fillna('Inconnu')
                    
        #             st.info("✅ Valeurs manquantes traitées")
                
        #         # Suppression des valeurs aberrantes
        #         if remove_outliers:
        #             # Suppression des prix négatifs ou très élevés
        #             q99 = processed_data['item_price'].quantile(0.99)
        #             initial_rows = len(processed_data)
        #             processed_data = processed_data[
        #                 (processed_data['item_price'] > 0) & 
        #                 (processed_data['item_price'] <= q99)
        #             ]
                    
        #             # Suppression des quantités négatives ou très élevées
        #             q99_qty = processed_data['item_cnt_day'].quantile(0.99)
        #             processed_data = processed_data[
        #                 (processed_data['item_cnt_day'] >= 0) & 
        #                 (processed_data['item_cnt_day'] <= q99_qty)
        #             ]
                    
        #             st.info(f"✅ {initial_rows - len(processed_data):,} valeurs aberrantes supprimées")
                
        #         # Agrégation mensuelle
        #         if aggregate_monthly:
        #             monthly_data = processed_data.groupby(['shop_id', 'item_id', 'date_block_num']).agg({
        #                 'item_cnt_day': 'sum',
        #                 'item_price': 'mean',
        #                 'item_category_id': 'first',
        #                 'item_name': 'first',
        #                 'shop_name': 'first',
        #                 'item_category_name': 'first'
        #             }).reset_index()
                    
        #             monthly_data = monthly_data.rename(columns={'item_cnt_day': 'item_cnt_month'})
                    
        #             # Filtrage par seuil de ventes
        #             if min_sales_threshold > 0:
        #                 initial_rows = len(monthly_data)
        #                 monthly_data = monthly_data[monthly_data['item_cnt_month'] >= min_sales_threshold]
        #                 st.info(f"✅ {initial_rows - len(monthly_data):,} lignes avec ventes < {min_sales_threshold} supprimées")
                    
        #             processed_data = monthly_data
        #             st.info("✅ Données agrégées en ventes mensuelles")
                
        #         # Sauvegarde des données prétraitées
        #         st.session_state['processed_data'] = processed_data
        #         st.success("🎉 Prétraitement terminé avec succès !")
        
        # Affichage des résultats si disponibles
        if 'processed_data' in st.session_state:
            processed_data = st.session_state['processed_data']
            
            st.subheader("📈 Résultats du prétraitement")
            
            # Comparaison avant/après
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Avant prétraitement**")
                st.metric("Lignes", f"{len(data):,}")
                st.metric("Valeurs manquantes", f"{data.isnull().sum().sum():,}")
                st.metric("Doublons", f"{data.duplicated().sum():,}")
                if 'item_cnt_day' in data.columns:
                    st.metric("Ventes totales", f"{data['item_cnt_day'].sum():,.0f}")
            
            with col2:
                st.markdown("**Après prétraitement**")
                st.metric("Lignes", f"{len(processed_data):,}")
                st.metric("Valeurs manquantes", f"{processed_data.isnull().sum().sum():,}")
                st.metric("Doublons", f"{processed_data.duplicated().sum():,}")
                sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
                if sales_col in processed_data.columns:
                    st.metric("Ventes totales", f"{processed_data[sales_col].sum():,.0f}")
            
            # Aperçu des données prétraitées
            st.subheader("🔍 Aperçu des données prétraitées")
            st.dataframe(processed_data.head(10))
            
            # Statistiques descriptives
            st.subheader("📊 Statistiques descriptives")
            
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                st.dataframe(processed_data[numeric_columns].describe())
            
            # Graphiques de distribution
            st.subheader("📈 Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'item_price' in processed_data.columns:
                    fig_price = px.histogram(
                        processed_data, 
                        x='item_price', 
                        title="Distribution des prix",
                        nbins=50
                    )
                    fig_price.update_layout(height=400)
                    st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
                if sales_col in processed_data.columns:
                    fig_sales = px.histogram(
                        processed_data, 
                        x=sales_col, 
                        title=f"Distribution des ventes ({'mensuelles' if sales_col == 'item_cnt_month' else 'quotidiennes'})",
                        nbins=50
                    )
                    fig_sales.update_layout(height=400)
                    st.plotly_chart(fig_sales, use_container_width=True)
            
            # Évolution temporelle
            if 'date_block_num' in processed_data.columns:
                st.subheader("📅 Évolution temporelle")
                
                sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
                temporal_data = processed_data.groupby('date_block_num')[sales_col].sum().reset_index()
                
                fig_temporal = px.line(
                    temporal_data,
                    x='date_block_num',
                    y=sales_col,
                    title="Évolution des ventes totales dans le temps",
                    markers=True
                )
                fig_temporal.update_layout(height=400)
                st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Top magasins et produits
            col1, col2 = st.columns(2)
            
            with col1:
                if 'shop_name' in processed_data.columns:
                    sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
                    top_shops = processed_data.groupby('shop_name')[sales_col].sum().sort_values(ascending=False).head(10)
                    
                    fig_shops = px.bar(
                        x=top_shops.values,
                        y=top_shops.index,
                        orientation='h',
                        title="Top 10 des magasins par ventes"
                    )
                    fig_shops.update_layout(height=400)
                    st.plotly_chart(fig_shops, use_container_width=True)
            
            with col2:
                if 'item_category_name' in processed_data.columns:
                    sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
                    top_categories = processed_data.groupby('item_category_name')[sales_col].sum().sort_values(ascending=False).head(10)
                    
                    fig_categories = px.bar(
                        x=top_categories.values,
                        y=top_categories.index,
                        orientation='h',
                        title="Top 10 des catégories par ventes"
                    )
                    fig_categories.update_layout(height=400)
                    st.plotly_chart(fig_categories, use_container_width=True)
            
            # Bouton de téléchargement
            st.subheader("💾 Téléchargement")
            
            csv_buffer = io.StringIO()
            processed_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Télécharger les données prétraitées (CSV)",
                data=csv_data,
                file_name=f"donnees_pretraitees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def show_modeling(self):
        """Interface de modélisation ML"""
        st.header("🤖 Modélisation Machine Learning")
        
        if 'sales_data' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données dans la section 'Import des Données'")
            return
        
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord prétraiter vos données dans la section 'Prétraitement'")
            return
        
        processed_data = st.session_state['processed_data']
        
        st.subheader("📊 Préparation des données pour l'entraînement")
        
        # Vérification des colonnes nécessaires
        sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
        
        if sales_col not in processed_data.columns:
            st.error("❌ Colonne de ventes manquante. Veuillez prétraiter vos données.")
            return
        
        # Configuration des paramètres d'entraînement
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Paramètres d'entraînement")
            
            # Sélection de la période d'entraînement
            max_date_block = processed_data['date_block_num'].max()
            min_date_block = processed_data['date_block_num'].min()
            
            train_end_month = st.slider(
                "Mois de fin d'entraînement",
                min_value=min_date_block + 6,
                max_value=max_date_block - 1,
                value=max_date_block - 3,
                help="Les données après ce mois seront utilisées pour la validation"
            )
            
            # Sélection des features
            st.markdown("**Features à utiliser:**")
            use_lag_features = st.checkbox("Features de lag (ventes précédentes)", value=True)
            use_price_features = st.checkbox("Features de prix", value=True)
            use_category_features = st.checkbox("Features de catégorie", value=True)
            use_shop_features = st.checkbox("Features de magasin", value=True)
            
            # Paramètres des modèles
            st.markdown("**Paramètres des modèles:**")
            rf_n_estimators = st.slider("Random Forest - Nombre d'arbres", 50, 500, 100)
            xgb_n_estimators = st.slider("XGBoost - Nombre d'arbres", 50, 500, 100)
        
        with col2:
            st.subheader("📈 Statistiques des données")
            
            # Statistiques générales
            st.metric("Nombre total d'observations", f"{len(processed_data):,}")
            st.metric("Période couverte (mois)", f"{processed_data['date_block_num'].nunique()}")
            st.metric("Nombre de magasins", f"{processed_data['shop_id'].nunique()}")
            st.metric("Nombre de produits", f"{processed_data['item_id'].nunique()}")
            
            # Distribution des ventes
            st.markdown("**Distribution des ventes:**")
            st.metric("Ventes moyennes", f"{processed_data[sales_col].mean():.2f}")
            st.metric("Ventes médianes", f"{processed_data[sales_col].median():.2f}")
            st.metric("Écart-type", f"{processed_data[sales_col].std():.2f}")
        
        # Bouton d'entraînement
        if st.button("🚀 Entraîner les modèles", type="primary"):
            with st.spinner("Préparation des features et entraînement en cours..."):
                
                # Préparation des features
                try:
                    # Création des features de base
                    model_data = processed_data.copy()
                    
                    # Tri par shop, item et date pour les features de lag
                    model_data = model_data.sort_values(['shop_id', 'item_id', 'date_block_num'])
                    
                    # Features de lag si demandées
                    if use_lag_features:
                        for lag in [1, 2, 3]:
                            model_data[f'{sales_col}_lag_{lag}'] = model_data.groupby(['shop_id', 'item_id'])[sales_col].shift(lag)
                        
                        # Moyennes mobiles
                        model_data[f'{sales_col}_ma_3'] = model_data.groupby(['shop_id', 'item_id'])[sales_col].rolling(3).mean().reset_index(0, drop=True)
                    
                    # Features agrégées par magasin
                    if use_shop_features:
                        shop_stats = model_data.groupby(['shop_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        shop_stats.columns = ['shop_id', 'date_block_num', 'shop_avg_sales', 'shop_std_sales']
                        model_data = model_data.merge(shop_stats, on=['shop_id', 'date_block_num'], how='left')
                    
                    # Features agrégées par catégorie
                    if use_category_features:
                        category_stats = model_data.groupby(['item_category_id', 'date_block_num'])[sales_col].agg(['mean', 'std']).reset_index()
                        category_stats.columns = ['item_category_id', 'date_block_num', 'category_avg_sales', 'category_std_sales']
                        model_data = model_data.merge(category_stats, on=['item_category_id', 'date_block_num'], how='left')
                    
                    # Sélection des features finales
                    feature_columns = ['shop_id', 'item_id', 'item_category_id', 'date_block_num']
                    
                    if use_price_features and 'item_price' in model_data.columns:
                        feature_columns.append('item_price')
                    
                    if use_lag_features:
                        lag_cols = [col for col in model_data.columns if 'lag' in col or 'ma_' in col]
                        feature_columns.extend(lag_cols)
                    
                    if use_shop_features:
                        shop_cols = [col for col in model_data.columns if 'shop_avg' in col or 'shop_std' in col]
                        feature_columns.extend(shop_cols)
                    
                    if use_category_features:
                        cat_cols = [col for col in model_data.columns if 'category_avg' in col or 'category_std' in col]
                        feature_columns.extend(cat_cols)
                    
                    # Suppression des lignes avec des valeurs manquantes (dues aux lags)
                    model_data = model_data.dropna()
                    
                    # Division train/validation
                    train_data = model_data[model_data['date_block_num'] <= train_end_month]
                    val_data = model_data[model_data['date_block_num'] > train_end_month]
                    
                    if len(train_data) == 0 or len(val_data) == 0:
                        st.error("❌ Pas assez de données pour l'entraînement ou la validation")
                        return
                    
                    X_train = train_data[feature_columns]
                    y_train = train_data[sales_col]
                    X_val = val_data[feature_columns]
                    y_val = val_data[sales_col]
                    
                    st.success(f"✅ Données préparées: {len(X_train):,} échantillons d'entraînement, {len(X_val):,} de validation")
                    
                    # Entraînement des modèles
                    models = {}
                    results = {}
                    
                    # Linear Regression
                    st.info("Entraînement du modèle Linear Regression...")
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    lr_pred = lr.predict(X_val)
                    
                    models['Linear Regression'] = lr
                    results['Linear Regression'] = {
                        'mae': mean_absolute_error(y_val, lr_pred),
                        'rmse': np.sqrt(mean_squared_error(y_val, lr_pred)),
                        'r2': r2_score(y_val, lr_pred),
                        'predictions': lr_pred
                    }
                    
                    # Random Forest
                    st.info("Entraînement du modèle Random Forest...")
                    rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=42, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    rf_pred = rf.predict(X_val)
                    
                    models['Random Forest'] = rf
                    results['Random Forest'] = {
                        'mae': mean_absolute_error(y_val, rf_pred),
                        'rmse': np.sqrt(mean_squared_error(y_val, rf_pred)),
                        'r2': r2_score(y_val, rf_pred),
                        'predictions': rf_pred
                    }
                    
                    # XGBoost
                    st.info("Entraînement du modèle XGBoost...")
                    xgb_model = xgb.XGBRegressor(n_estimators=xgb_n_estimators, random_state=42, n_jobs=-1)
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_val)
                    
                    models['XGBoost'] = xgb_model
                    results['XGBoost'] = {
                        'mae': mean_absolute_error(y_val, xgb_pred),
                        'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred)),
                        'r2': r2_score(y_val, xgb_pred),
                        'predictions': xgb_pred
                    }
                    
                    # Sauvegarde des résultats
                    st.session_state['trained_models'] = models
                    st.session_state['model_results'] = results
                    st.session_state['validation_data'] = {'X_val': X_val, 'y_val': y_val}
                    st.session_state['feature_columns'] = feature_columns
                    
                    st.success("🎉 Entraînement terminé avec succès !")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")
                    return
        
        # Affichage des résultats si disponibles
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            
            st.subheader("📊 Résultats des modèles")
            
            # Tableau de comparaison des performances
            performance_df = pd.DataFrame({
                'Modèle': list(results.keys()),
                'MAE': [results[model]['mae'] for model in results.keys()],
                'RMSE': [results[model]['rmse'] for model in results.keys()],
                'R²': [results[model]['r2'] for model in results.keys()]
            })
            
            # Tri par R² décroissant
            performance_df = performance_df.sort_values('R²', ascending=False)
            
            st.dataframe(performance_df, use_container_width=True)
            
            # Graphique de comparaison des métriques
            col1, col2 = st.columns(2)
            
            with col1:
                fig_metrics = px.bar(
                    performance_df,
                    x='Modèle',
                    y='R²',
                    title="Comparaison du R² des modèles",
                    color='R²',
                    color_continuous_scale='viridis'
                )
                fig_metrics.update_layout(height=400)
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            with col2:
                fig_mae = px.bar(
                    performance_df,
                    x='Modèle',
                    y='MAE',
                    title="Comparaison du MAE des modèles",
                    color='MAE',
                    color_continuous_scale='viridis_r'
                )
                fig_mae.update_layout(height=400)
                st.plotly_chart(fig_mae, use_container_width=True)
            
            # Graphiques de prédictions vs réalité
            st.subheader("🎯 Prédictions vs Réalité")
            
            val_data = st.session_state['validation_data']
            y_val = val_data['y_val']
            
            # Sélection du modèle à visualiser
            selected_model = st.selectbox(
                "Choisir un modèle à visualiser",
                list(results.keys()),
                index=0
            )
            
            predictions = results[selected_model]['predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot prédictions vs réalité
                scatter_data = pd.DataFrame({
                    'Réel': y_val,
                    'Prédit': predictions
                })
                
                fig_scatter = px.scatter(
                    scatter_data,
                    x='Réel',
                    y='Prédit',
                    title=f"Prédictions vs Réalité - {selected_model}",
                    opacity=0.6
                )
                
                # Ligne de référence y=x
                min_val = min(y_val.min(), predictions.min())
                max_val = max(y_val.max(), predictions.max())
                fig_scatter.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Distribution des erreurs
                errors = predictions - y_val
                
                fig_errors = px.histogram(
                    x=errors,
                    title=f"Distribution des erreurs - {selected_model}",
                    nbins=50
                )
                fig_errors.update_layout(height=400)
                st.plotly_chart(fig_errors, use_container_width=True)
            
            # Importance des features (pour Random Forest et XGBoost)
            if selected_model in ['Random Forest', 'XGBoost']:
                st.subheader("🔍 Importance des features")
                
                model = st.session_state['trained_models'][selected_model]
                feature_names = st.session_state['feature_columns']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 des features importantes - {selected_model}"
                )
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Métriques détaillées
            st.subheader("📈 Métriques détaillées")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mean Absolute Error (MAE)",
                    f"{results[selected_model]['mae']:.3f}",
                    help="Erreur absolue moyenne - plus c'est bas, mieux c'est"
                )
            
            with col2:
                st.metric(
                    "Root Mean Square Error (RMSE)",
                    f"{results[selected_model]['rmse']:.3f}",
                    help="Racine de l'erreur quadratique moyenne - plus c'est bas, mieux c'est"
                )
            
            with col3:
                st.metric(
                    "Coefficient de détermination (R²)",
                    f"{results[selected_model]['r2']:.3f}",
                    help="Proportion de variance expliquée - plus c'est proche de 1, mieux c'est"
                )
            
            # Recommandations
            st.subheader("💡 Recommandations")
            
            best_model = performance_df.iloc[0]['Modèle']
            best_r2 = performance_df.iloc[0]['R²']
            
            if best_r2 > 0.8:
                st.success(f"✅ Excellent ! Le modèle {best_model} a un R² de {best_r2:.3f}, ce qui indique une très bonne capacité prédictive.")
            elif best_r2 > 0.6:
                st.info(f"👍 Bon résultat ! Le modèle {best_model} a un R² de {best_r2:.3f}. Vous pourriez améliorer en ajoutant plus de features ou de données.")
            elif best_r2 > 0.4:
                st.warning(f"⚠️ Résultat moyen. Le modèle {best_model} a un R² de {best_r2:.3f}. Considérez l'ajout de plus de features ou l'optimisation des hyperparamètres.")
            else:
                st.error(f"❌ Résultat faible. Le modèle {best_model} a un R² de {best_r2:.3f}. Les données pourraient nécessiter plus de preprocessing ou de features.")
    
    def show_visualizations(self):
        """Interface de visualisations"""
        st.header("📈 Visualisations")
        
        if 'sales_data' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données dans la section 'Import des Données'")
            return
        
        data = st.session_state.get('processed_data', st.session_state['sales_data'])
        sales_col = 'item_cnt_month' if 'item_cnt_month' in data.columns else 'item_cnt_day'
        
        # Sidebar pour les filtres
        st.sidebar.header("🔍 Filtres")
        
        # Filtre par période
        if 'date_block_num' in data.columns:
            date_range = st.sidebar.slider(
                "Période à analyser",
                min_value=int(data['date_block_num'].min()),
                max_value=int(data['date_block_num'].max()),
                value=(int(data['date_block_num'].min()), int(data['date_block_num'].max()))
            )
            filtered_data = data[
                (data['date_block_num'] >= date_range[0]) & 
                (data['date_block_num'] <= date_range[1])
            ]
        else:
            filtered_data = data
        
        # Filtre par magasin
        if 'shop_name' in filtered_data.columns:
            selected_shops = st.sidebar.multiselect(
                "Magasins à analyser",
                options=sorted(filtered_data['shop_name'].unique()),
                default=sorted(filtered_data['shop_name'].unique())[:5] if len(filtered_data['shop_name'].unique()) > 5 else sorted(filtered_data['shop_name'].unique())
            )
            if selected_shops:
                filtered_data = filtered_data[filtered_data['shop_name'].isin(selected_shops)]
        
        # Filtre par catégorie
        if 'item_category_name' in filtered_data.columns:
            selected_categories = st.sidebar.multiselect(
                "Catégories à analyser",
                options=sorted(filtered_data['item_category_name'].unique()),
                default=sorted(filtered_data['item_category_name'].unique())[:5] if len(filtered_data['item_category_name'].unique()) > 5 else sorted(filtered_data['item_category_name'].unique())
            )
            if selected_categories:
                filtered_data = filtered_data[filtered_data['item_category_name'].isin(selected_categories)]
        
        # Onglets pour différents types de visualisations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vue d'ensemble", 
            "📈 Tendances temporelles", 
            "🏪 Analyse par magasin", 
            "📦 Analyse par produit", 
            "🎯 Prédictions"
        ])
        
        with tab1:
            st.subheader("📊 Vue d'ensemble des ventes")
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = filtered_data[sales_col].sum()
                st.metric("Ventes totales", f"{total_sales:,.0f}")
            
            with col2:
                avg_sales = filtered_data[sales_col].mean()
                st.metric("Ventes moyennes", f"{avg_sales:.2f}")
            
            with col3:
                unique_items = filtered_data['item_id'].nunique() if 'item_id' in filtered_data.columns else 0
                st.metric("Produits uniques", f"{unique_items:,}")
            
            with col4:
                unique_shops = filtered_data['shop_id'].nunique() if 'shop_id' in filtered_data.columns else 0
                st.metric("Magasins actifs", f"{unique_shops:,}")
            
            # Distribution des ventes
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    filtered_data,
                    x=sales_col,
                    title="Distribution des ventes",
                    nbins=50,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot des ventes par catégorie (si disponible)
                if 'item_category_name' in filtered_data.columns and len(selected_categories) <= 10:
                    fig_box = px.box(
                        filtered_data,
                        x='item_category_name',
                        y=sales_col,
                        title="Distribution des ventes par catégorie"
                    )
                    fig_box.update_xaxes(tickangle=45)
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    # Graphique en secteurs des ventes par catégorie
                    if 'item_category_name' in filtered_data.columns:
                        category_sales = filtered_data.groupby('item_category_name')[sales_col].sum().sort_values(ascending=False).head(10)
                        fig_pie = px.pie(
                            values=category_sales.values,
                            names=category_sales.index,
                            title="Top 10 des catégories par ventes"
                        )
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            # Heatmap des ventes (si données temporelles disponibles)
            if 'date_block_num' in filtered_data.columns and 'shop_id' in filtered_data.columns:
                st.subheader("🔥 Heatmap des ventes par magasin et période")
                
                # Agrégation pour la heatmap
                heatmap_data = filtered_data.groupby(['shop_id', 'date_block_num'])[sales_col].sum().reset_index()
                heatmap_pivot = heatmap_data.pivot(index='shop_id', columns='date_block_num', values=sales_col).fillna(0)
                
                # Limitation à 20 magasins max pour la lisibilité
                if len(heatmap_pivot) > 20:
                    top_shops = filtered_data.groupby('shop_id')[sales_col].sum().sort_values(ascending=False).head(20).index
                    heatmap_pivot = heatmap_pivot.loc[top_shops]
                
                fig_heatmap = px.imshow(
                    heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    title="Ventes par magasin et période",
                    color_continuous_scale='viridis',
                    aspect='auto'
                )
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Analyse des tendances temporelles")
            
            if 'date_block_num' not in filtered_data.columns:
                st.warning("⚠️ Données temporelles non disponibles")
            else:
                # Évolution des ventes totales
                temporal_data = filtered_data.groupby('date_block_num')[sales_col].agg(['sum', 'mean', 'count']).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_total = px.line(
                        temporal_data,
                        x='date_block_num',
                        y='sum',
                        title="Évolution des ventes totales",
                        markers=True,
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_total.update_layout(height=400)
                    st.plotly_chart(fig_total, use_container_width=True)
                
                with col2:
                    fig_avg = px.line(
                        temporal_data,
                        x='date_block_num',
                        y='mean',
                        title="Évolution des ventes moyennes",
                        markers=True,
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_avg.update_layout(height=400)
                    st.plotly_chart(fig_avg, use_container_width=True)
                
                # Tendances par magasin (top 5)
                if 'shop_name' in filtered_data.columns:
                    st.subheader("🏪 Tendances par magasin (Top 5)")
                    
                    top_shops_sales = filtered_data.groupby('shop_name')[sales_col].sum().sort_values(ascending=False).head(5)
                    shop_temporal = filtered_data[filtered_data['shop_name'].isin(top_shops_sales.index)]
                    shop_temporal = shop_temporal.groupby(['shop_name', 'date_block_num'])[sales_col].sum().reset_index()
                    
                    fig_shops = px.line(
                        shop_temporal,
                        x='date_block_num',
                        y=sales_col,
                        color='shop_name',
                        title="Évolution des ventes par magasin (Top 5)",
                        markers=True
                    )
                    fig_shops.update_layout(height=500)
                    st.plotly_chart(fig_shops, use_container_width=True)
                
                # Tendances par catégorie (top 5)
                if 'item_category_name' in filtered_data.columns:
                    st.subheader("📦 Tendances par catégorie (Top 5)")
                    
                    top_categories_sales = filtered_data.groupby('item_category_name')[sales_col].sum().sort_values(ascending=False).head(5)
                    category_temporal = filtered_data[filtered_data['item_category_name'].isin(top_categories_sales.index)]
                    category_temporal = category_temporal.groupby(['item_category_name', 'date_block_num'])[sales_col].sum().reset_index()
                    
                    fig_categories = px.line(
                        category_temporal,
                        x='date_block_num',
                        y=sales_col,
                        color='item_category_name',
                        title="Évolution des ventes par catégorie (Top 5)",
                        markers=True
                    )
                    fig_categories.update_layout(height=500)
                    st.plotly_chart(fig_categories, use_container_width=True)
                
                # Analyse de saisonnalité
                st.subheader("🌊 Analyse de saisonnalité")
                
                # Calcul des moyennes par mois (modulo 12)
                if len(temporal_data) >= 12:
                    temporal_data['month'] = temporal_data['date_block_num'] % 12
                    seasonal_data = temporal_data.groupby('month')['sum'].mean().reset_index()
                    seasonal_data['month_name'] = seasonal_data['month'].map({
                        0: 'Jan', 1: 'Fév', 2: 'Mar', 3: 'Avr', 4: 'Mai', 5: 'Jun',
                        6: 'Jul', 7: 'Aoû', 8: 'Sep', 9: 'Oct', 10: 'Nov', 11: 'Déc'
                    })
                    
                    fig_seasonal = px.bar(
                        seasonal_data,
                        x='month_name',
                        y='sum',
                        title="Saisonnalité des ventes (moyenne par mois)",
                        color='sum',
                        color_continuous_scale='viridis'
                    )
                    fig_seasonal.update_layout(height=400)
                    st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with tab3:
            st.subheader("🏪 Analyse par magasin")
            
            if 'shop_name' not in filtered_data.columns:
                st.warning("⚠️ Données de magasins non disponibles")
            else:
                # Performance des magasins
                shop_performance = filtered_data.groupby('shop_name').agg({
                    sales_col: ['sum', 'mean', 'count'],
                    'item_id': 'nunique' if 'item_id' in filtered_data.columns else lambda x: 0
                }).round(2)
                
                shop_performance.columns = ['Ventes totales', 'Ventes moyennes', 'Nombre de transactions', 'Produits uniques']
                shop_performance = shop_performance.sort_values('Ventes totales', ascending=False)
                
                # Top 10 des magasins
                col1, col2 = st.columns(2)
                
                with col1:
                    top_10_shops = shop_performance.head(10)
                    fig_top_shops = px.bar(
                        x=top_10_shops['Ventes totales'],
                        y=top_10_shops.index,
                        orientation='h',
                        title="Top 10 des magasins par ventes totales",
                        color=top_10_shops['Ventes totales'],
                        color_continuous_scale='viridis'
                    )
                    fig_top_shops.update_layout(height=500)
                    st.plotly_chart(fig_top_shops, use_container_width=True)
                
                with col2:
                    fig_avg_shops = px.bar(
                        x=top_10_shops['Ventes moyennes'],
                        y=top_10_shops.index,
                        orientation='h',
                        title="Top 10 des magasins par ventes moyennes",
                        color=top_10_shops['Ventes moyennes'],
                        color_continuous_scale='plasma'
                    )
                    fig_avg_shops.update_layout(height=500)
                    st.plotly_chart(fig_avg_shops, use_container_width=True)
                
                # Tableau détaillé
                st.subheader("📋 Performance détaillée des magasins")
                st.dataframe(shop_performance, use_container_width=True)
                
                # Analyse de corrélation prix-ventes par magasin
                if 'item_price' in filtered_data.columns:
                    st.subheader("💰 Relation prix-ventes par magasin")
                    
                    price_sales_corr = filtered_data.groupby('shop_name').agg({
                        'item_price': 'mean',
                        sales_col: 'sum'
                    }).reset_index()
                    
                    fig_price_corr = px.scatter(
                        price_sales_corr,
                        x='item_price',
                        y=sales_col,
                        hover_data=['shop_name'],
                        title="Corrélation entre prix moyen et ventes totales par magasin",
                        trendline="ols"
                    )
                    fig_price_corr.update_layout(height=500)
                    st.plotly_chart(fig_price_corr, use_container_width=True)
        
        with tab4:
            st.subheader("📦 Analyse par produit et catégorie")
            
            # Performance des catégories
            if 'item_category_name' in filtered_data.columns:
                category_performance = filtered_data.groupby('item_category_name').agg({
                    sales_col: ['sum', 'mean', 'count'],
                    'item_id': 'nunique' if 'item_id' in filtered_data.columns else lambda x: 0,
                    'item_price': 'mean' if 'item_price' in filtered_data.columns else lambda x: 0
                }).round(2)
                
                category_performance.columns = ['Ventes totales', 'Ventes moyennes', 'Transactions', 'Produits uniques', 'Prix moyen']
                category_performance = category_performance.sort_values('Ventes totales', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    top_categories = category_performance.head(10)
                    fig_cat_total = px.bar(
                        x=top_categories.index,
                        y=top_categories['Ventes totales'],
                        title="Top 10 des catégories par ventes totales",
                        color=top_categories['Ventes totales'],
                        color_continuous_scale='viridis'
                    )
                    fig_cat_total.update_xaxes(tickangle=45)
                    fig_cat_total.update_layout(height=500)
                    st.plotly_chart(fig_cat_total, use_container_width=True)
                
                with col2:
                    fig_cat_avg = px.bar(
                        x=top_categories.index,
                        y=top_categories['Ventes moyennes'],
                        title="Top 10 des catégories par ventes moyennes",
                        color=top_categories['Ventes moyennes'],
                        color_continuous_scale='plasma'
                    )
                    fig_cat_avg.update_xaxes(tickangle=45)
                    fig_cat_avg.update_layout(height=500)
                    st.plotly_chart(fig_cat_avg, use_container_width=True)
                
                # Tableau des catégories
                st.subheader("📊 Performance des catégories")
                st.dataframe(category_performance, use_container_width=True)
            
            # Top produits
            if 'item_name' in filtered_data.columns:
                st.subheader("🏆 Top produits")
                
                product_performance = filtered_data.groupby('item_name')[sales_col].sum().sort_values(ascending=False).head(20)
                
                fig_products = px.bar(
                    x=product_performance.values,
                    y=product_performance.index,
                    orientation='h',
                    title="Top 20 des produits par ventes",
                    color=product_performance.values,
                    color_continuous_scale='viridis'
                )
                fig_products.update_layout(height=600)
                st.plotly_chart(fig_products, use_container_width=True)
            
            # Analyse ABC des produits
            if 'item_id' in filtered_data.columns:
                st.subheader("📈 Analyse ABC des produits")
                
                product_sales = filtered_data.groupby('item_id')[sales_col].sum().sort_values(ascending=False)
                total_sales = product_sales.sum()
                
                # Calcul des pourcentages cumulés
                cumulative_pct = (product_sales.cumsum() / total_sales * 100)
                
                # Classification ABC
                abc_data = pd.DataFrame({
                    'Produit': range(1, len(product_sales) + 1),
                    'Ventes': product_sales.values,
                    'Pourcentage cumulé': cumulative_pct.values
                })
                
                fig_abc = px.line(
                    abc_data,
                    x='Produit',
                    y='Pourcentage cumulé',
                    title="Courbe ABC des produits (Loi de Pareto)",
                    markers=True
                )
                
                # Lignes de référence pour A (80%), B (95%)
                fig_abc.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Classe A (80%)")
                fig_abc.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="Classe B (95%)")
                
                fig_abc.update_layout(height=500)
                st.plotly_chart(fig_abc, use_container_width=True)
                
                # Statistiques ABC
                class_a = (cumulative_pct <= 80).sum()
                class_b = ((cumulative_pct > 80) & (cumulative_pct <= 95)).sum()
                class_c = (cumulative_pct > 95).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classe A (80% des ventes)", f"{class_a} produits ({class_a/len(product_sales)*100:.1f}%)")
                with col2:
                    st.metric("Classe B (15% des ventes)", f"{class_b} produits ({class_b/len(product_sales)*100:.1f}%)")
                with col3:
                    st.metric("Classe C (5% des ventes)", f"{class_c} produits ({class_c/len(product_sales)*100:.1f}%)")
        
        with tab5:
            st.subheader("🎯 Visualisation des prédictions")
            
            if 'trained_models' not in st.session_state:
                st.info("ℹ️ Aucun modèle entraîné. Veuillez d'abord entraîner des modèles dans la section 'Modélisation'.")
            else:
                models = st.session_state['trained_models']
                results = st.session_state['model_results']
                
                # Sélection du modèle
                selected_model = st.selectbox(
                    "Choisir un modèle pour les visualisations",
                    list(models.keys())
                )
                
                predictions = results[selected_model]['predictions']
                val_data = st.session_state['validation_data']
                y_val = val_data['y_val']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique de prédictions vs réalité
                    pred_real_data = pd.DataFrame({
                        'Réel': y_val,
                        'Prédit': predictions,
                        'Erreur': predictions - y_val
                    })
                    
                    fig_pred = px.scatter(
                        pred_real_data,
                        x='Réel',
                        y='Prédit',
                        color='Erreur',
                        title=f"Prédictions vs Réalité - {selected_model}",
                        color_continuous_scale='RdYlBu_r',
                        hover_data=['Erreur']
                    )
                    
                    # Ligne de référence parfaite
                    min_val = min(y_val.min(), predictions.min())
                    max_val = max(y_val.max(), predictions.max())
                    fig_pred.add_shape(
                        type="line",
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(color="red", dash="dash", width=2)
                    )
                    
                    fig_pred.update_layout(height=500)
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # Distribution des erreurs
                    fig_errors = px.histogram(
                        pred_real_data,
                        x='Erreur',
                        title=f"Distribution des erreurs - {selected_model}",
                        nbins=50,
                        color_discrete_sequence=['#ff7f0e']
                    )
                    
                    # Ligne verticale à 0
                    fig_errors.add_vline(x=0, line_dash="dash", line_color="red")
                    fig_errors.update_layout(height=500)
                    st.plotly_chart(fig_errors, use_container_width=True)
                
                # Métriques de performance
                st.subheader("📊 Métriques de performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mae = results[selected_model]['mae']
                    st.metric("MAE", f"{mae:.3f}")
                
                with col2:
                    rmse = results[selected_model]['rmse']
                    st.metric("RMSE", f"{rmse:.3f}")
                
                with col3:
                    r2 = results[selected_model]['r2']
                    st.metric("R²", f"{r2:.3f}")
                
                with col4:
                    mape = np.mean(np.abs((y_val - predictions) / np.maximum(y_val, 1))) * 100
                    st.metric("MAPE", f"{mape:.1f}%")
                
                # Analyse des résidus
                st.subheader("🔍 Analyse des résidus")
                
                residuals = y_val - predictions
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Q-Q plot des résidus
                    from scipy import stats
                    
                    qq_data = stats.probplot(residuals, dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode='markers',
                        name='Résidus observés'
                    )
                    fig_qq.add_scatter(
                        x=qq_data[0][0],
                        y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                        mode='lines',
                        name='Droite théorique',
                        line=dict(color='red', dash='dash')
                    )
                    fig_qq.update_layout(
                        title="Q-Q Plot des résidus",
                        xaxis_title="Quantiles théoriques",
                        yaxis_title="Quantiles observés",
                        height=400
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                with col2:
                    # Résidus vs prédictions
                    fig_residuals = px.scatter(
                        x=predictions,
                        y=residuals,
                        title="Résidus vs Prédictions",
                        labels={'x': 'Prédictions', 'y': 'Résidus'},
                        opacity=0.6
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residuals.update_layout(height=400)
                    st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Bouton de téléchargement des visualisations
        st.subheader("💾 Export des données filtrées")
        
        if st.button("📥 Télécharger les données filtrées"):
            csv_buffer = io.StringIO()
            filtered_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"donnees_filtrees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    def show_predictions(self):
        """Interface de prédictions"""
        st.header("📊 Prédictions")
        
        if 'sales_data' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données dans la section 'Import des Données'")
            return
        
        if 'trained_models' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord entraîner des modèles dans la section 'Modélisation'")
            return
        
        models = st.session_state['trained_models']
        processed_data = st.session_state.get('processed_data', st.session_state['sales_data'])
        sales_col = 'item_cnt_month' if 'item_cnt_month' in processed_data.columns else 'item_cnt_day'
        
        # Onglets pour différents types de prédictions
        tab1, tab2, tab3 = st.tabs([
            "🎯 Prédictions personnalisées", 
            "📈 Prédictions en lot", 
            "🔮 Prédictions futures"
        ])
        
        with tab1:
            st.subheader("🎯 Prédictions personnalisées")
            st.markdown("Faites des prédictions pour des combinaisons spécifiques magasin-produit.")
            
            # Sélection du modèle
            selected_model_name = st.selectbox(
                "Choisir le modèle de prédiction",
                list(models.keys()),
                help="Sélectionnez le modèle entraîné à utiliser pour les prédictions"
            )
            
            selected_model = models[selected_model_name]
            feature_columns = st.session_state['feature_columns']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Paramètres de prédiction:**")
                
                # Sélection du magasin
                if 'shop_id' in processed_data.columns:
                    available_shops = processed_data[['shop_id', 'shop_name']].drop_duplicates() if 'shop_name' in processed_data.columns else processed_data[['shop_id']].drop_duplicates()
                    
                    if 'shop_name' in processed_data.columns:
                        shop_options = {f"{row['shop_name']} (ID: {row['shop_id']})": row['shop_id'] for _, row in available_shops.iterrows()}
                        selected_shop_display = st.selectbox("Magasin", list(shop_options.keys()))
                        selected_shop_id = shop_options[selected_shop_display]
                    else:
                        selected_shop_id = st.selectbox("Shop ID", available_shops['shop_id'].tolist())
                
                # Sélection du produit
                if 'item_id' in processed_data.columns:
                    # Filtrer les produits disponibles dans le magasin sélectionné
                    shop_items = processed_data[processed_data['shop_id'] == selected_shop_id]
                    available_items = shop_items[['item_id', 'item_name']].drop_duplicates() if 'item_name' in processed_data.columns else shop_items[['item_id']].drop_duplicates()
                    
                    if 'item_name' in processed_data.columns and len(available_items) > 0:
                        item_options = {f"{row['item_name']} (ID: {row['item_id']})": row['item_id'] for _, row in available_items.iterrows()}
                        if item_options:
                            selected_item_display = st.selectbox("Produit", list(item_options.keys()))
                            selected_item_id = item_options[selected_item_display]
                        else:
                            st.warning("Aucun produit trouvé pour ce magasin")
                            selected_item_id = None
                    else:
                        if len(available_items) > 0:
                            selected_item_id = st.selectbox("Item ID", available_items['item_id'].tolist())
                        else:
                            st.warning("Aucun produit trouvé pour ce magasin")
                            selected_item_id = None
                
                # Période de prédiction
                max_date_block = processed_data['date_block_num'].max()
                prediction_month = st.number_input(
                    "Mois de prédiction",
                    min_value=max_date_block + 1,
                    max_value=max_date_block + 12,
                    value=max_date_block + 1,
                    help="Numéro du mois pour lequel faire la prédiction"
                )
                
                # Prix du produit (si nécessaire)
                if 'item_price' in feature_columns:
                    # Obtenir le prix historique moyen du produit
                    if selected_item_id is not None:
                        historical_prices = processed_data[
                            (processed_data['shop_id'] == selected_shop_id) & 
                            (processed_data['item_id'] == selected_item_id)
                        ]['item_price']
                        
                        if len(historical_prices) > 0:
                            default_price = historical_prices.mean()
                            price_range = (historical_prices.min(), historical_prices.max())
                        else:
                            default_price = processed_data['item_price'].mean()
                            price_range = (processed_data['item_price'].min(), processed_data['item_price'].max())
                    else:
                        default_price = processed_data['item_price'].mean()
                        price_range = (processed_data['item_price'].min(), processed_data['item_price'].max())
                    
                    item_price = st.number_input(
                        "Prix du produit",
                        min_value=float(price_range[0]),
                        max_value=float(price_range[1]),
                        value=float(default_price),
                        step=0.01,
                        help=f"Prix historique moyen: {default_price:.2f}"
                    )
            
            with col2:
                st.markdown("**Informations historiques:**")
                
                if selected_item_id is not None:
                    # Données historiques pour cette combinaison
                    historical_data = processed_data[
                        (processed_data['shop_id'] == selected_shop_id) & 
                        (processed_data['item_id'] == selected_item_id)
                    ].sort_values('date_block_num')
                    
                    if len(historical_data) > 0:
                        st.metric("Ventes historiques moyennes", f"{historical_data[sales_col].mean():.2f}")
                        st.metric("Ventes maximales", f"{historical_data[sales_col].max():.0f}")
                        st.metric("Dernière vente enregistrée", f"{historical_data[sales_col].iloc[-1]:.0f}")
                        st.metric("Nombre de mois avec données", f"{len(historical_data)}")
                        
                        # Graphique des ventes historiques
                        if len(historical_data) > 1:
                            fig_hist = px.line(
                                historical_data,
                                x='date_block_num',
                                y=sales_col,
                                title="Historique des ventes",
                                markers=True
                            )
                            fig_hist.update_layout(height=300)
                            st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Aucune donnée historique pour cette combinaison")
            
            # Bouton de prédiction
            if st.button("🔮 Faire la prédiction", type="primary"):
                if selected_item_id is not None:
                    try:
                        # Préparation des features pour la prédiction
                        prediction_features = {}
                        
                        # Features de base
                        prediction_features['shop_id'] = selected_shop_id
                        prediction_features['item_id'] = selected_item_id
                        prediction_features['date_block_num'] = prediction_month
                        
                        # Obtenir la catégorie du produit
                        item_category = processed_data[processed_data['item_id'] == selected_item_id]['item_category_id'].iloc[0]
                        prediction_features['item_category_id'] = item_category
                        
                        # Prix
                        if 'item_price' in feature_columns:
                            prediction_features['item_price'] = item_price
                        
                        # Features de lag (utiliser les dernières valeurs disponibles)
                        if any('lag' in col for col in feature_columns):
                            recent_data = processed_data[
                                (processed_data['shop_id'] == selected_shop_id) & 
                                (processed_data['item_id'] == selected_item_id)
                            ].sort_values('date_block_num').tail(3)
                            
                            for lag in [1, 2, 3]:
                                lag_col = f'{sales_col}_lag_{lag}'
                                if lag_col in feature_columns:
                                    if len(recent_data) >= lag:
                                        prediction_features[lag_col] = recent_data[sales_col].iloc[-lag]
                                    else:
                                        prediction_features[lag_col] = 0
                            
                            # Moyenne mobile
                            ma_col = f'{sales_col}_ma_3'
                            if ma_col in feature_columns:
                                if len(recent_data) >= 3:
                                    prediction_features[ma_col] = recent_data[sales_col].tail(3).mean()
                                else:
                                    prediction_features[ma_col] = recent_data[sales_col].mean() if len(recent_data) > 0 else 0
                        
                        # Features agrégées (utiliser les moyennes historiques)
                        if any('shop_avg' in col for col in feature_columns):
                            shop_avg = processed_data[processed_data['shop_id'] == selected_shop_id][sales_col].mean()
                            shop_std = processed_data[processed_data['shop_id'] == selected_shop_id][sales_col].std()
                            prediction_features['shop_avg_sales'] = shop_avg
                            prediction_features['shop_std_sales'] = shop_std
                        
                        if any('category_avg' in col for col in feature_columns):
                            cat_avg = processed_data[processed_data['item_category_id'] == item_category][sales_col].mean()
                            cat_std = processed_data[processed_data['item_category_id'] == item_category][sales_col].std()
                            prediction_features['category_avg_sales'] = cat_avg
                            prediction_features['category_std_sales'] = cat_std
                        
                        # Créer le DataFrame pour la prédiction
                        pred_df = pd.DataFrame([prediction_features])
                        
                        # S'assurer que toutes les colonnes nécessaires sont présentes
                        for col in feature_columns:
                            if col not in pred_df.columns:
                                pred_df[col] = 0
                        
                        pred_df = pred_df[feature_columns]
                        
                        # Faire la prédiction
                        prediction = selected_model.predict(pred_df)[0]
                        prediction = max(0, prediction)  # Pas de ventes négatives
                        
                        # Affichage du résultat
                        st.success("🎉 Prédiction réalisée avec succès !")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Ventes prédites",
                                f"{prediction:.1f}",
                                help=f"Prédiction pour le mois {prediction_month}"
                            )
                        
                        with col2:
                            # Comparaison avec la moyenne historique
                            if len(historical_data) > 0:
                                historical_avg = historical_data[sales_col].mean()
                                delta = prediction - historical_avg
                                st.metric(
                                    "vs Moyenne historique",
                                    f"{historical_avg:.1f}",
                                    delta=f"{delta:+.1f}"
                                )
                        
                        with col3:
                            # Intervalle de confiance approximatif
                            model_results = st.session_state['model_results']
                            rmse = model_results[selected_model_name]['rmse']
                            confidence_interval = 1.96 * rmse  # 95% de confiance
                            
                            st.metric(
                                "Intervalle de confiance (95%)",
                                f"[{max(0, prediction - confidence_interval):.1f}, {prediction + confidence_interval:.1f}]"
                            )
                        
                        # Sauvegarde de la prédiction
                        if 'custom_predictions' not in st.session_state:
                            st.session_state['custom_predictions'] = []
                        
                        st.session_state['custom_predictions'].append({
                            'timestamp': datetime.now(),
                            'model': selected_model_name,
                            'shop_id': selected_shop_id,
                            'item_id': selected_item_id,
                            'prediction_month': prediction_month,
                            'predicted_sales': prediction,
                            'item_price': item_price if 'item_price' in feature_columns else None
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
                else:
                    st.error("❌ Veuillez sélectionner un produit valide")
        
        with tab2:
            st.subheader("📈 Prédictions en lot")
            st.markdown("Générez des prédictions pour plusieurs combinaisons magasin-produit simultanément.")
            
            # Sélection du modèle
            batch_model_name = st.selectbox(
                "Modèle pour prédictions en lot",
                list(models.keys()),
                key="batch_model"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Paramètres de sélection:**")
                
                # Sélection des magasins
                if 'shop_name' in processed_data.columns:
                    available_shops = processed_data[['shop_id', 'shop_name']].drop_duplicates()
                    shop_options = {f"{row['shop_name']} (ID: {row['shop_id']})": row['shop_id'] for _, row in available_shops.iterrows()}
                    selected_shops_display = st.multiselect(
                        "Magasins à inclure",
                        list(shop_options.keys()),
                        default=list(shop_options.keys())[:5]
                    )
                    selected_shops = [shop_options[shop] for shop in selected_shops_display]
                else:
                    selected_shops = st.multiselect(
                        "Shop IDs",
                        processed_data['shop_id'].unique().tolist(),
                        default=processed_data['shop_id'].unique().tolist()[:5]
                    )
                
                # Critères de sélection des produits
                min_historical_sales = st.number_input(
                    "Ventes historiques minimales",
                    min_value=0,
                    value=10,
                    help="Inclure seulement les produits avec au moins ce nombre de ventes historiques"
                )
                
                prediction_month_batch = st.number_input(
                    "Mois de prédiction",
                    min_value=processed_data['date_block_num'].max() + 1,
                    max_value=processed_data['date_block_num'].max() + 12,
                    value=processed_data['date_block_num'].max() + 1,
                    key="batch_month"
                )
            
            with col2:
                st.markdown("**Aperçu de la sélection:**")
                
                # Calcul du nombre de combinaisons
                if selected_shops:
                    shop_item_combinations = processed_data[
                        processed_data['shop_id'].isin(selected_shops)
                    ].groupby(['shop_id', 'item_id'])[sales_col].sum()
                    
                    valid_combinations = shop_item_combinations[
                        shop_item_combinations >= min_historical_sales
                    ]
                    
                    st.metric("Magasins sélectionnés", len(selected_shops))
                    st.metric("Combinaisons valides", len(valid_combinations))
                    st.metric("Ventes totales historiques", f"{valid_combinations.sum():,.0f}")
            
            # Bouton de génération
            if st.button("🚀 Générer les prédictions en lot", type="primary"):
                if selected_shops:
                    with st.spinner("Génération des prédictions en cours..."):
                        try:
                            batch_model = models[batch_model_name]
                            feature_columns = st.session_state['feature_columns']
                            
                            # Préparer les données pour prédiction
                            prediction_data = []
                            
                            for shop_id in selected_shops:
                                shop_items = processed_data[
                                    processed_data['shop_id'] == shop_id
                                ].groupby('item_id')[sales_col].sum()
                                
                                valid_items = shop_items[shop_items >= min_historical_sales].index
                                
                                for item_id in valid_items:
                                    # Préparer les features comme dans la prédiction individuelle
                                    prediction_features = {
                                        'shop_id': shop_id,
                                        'item_id': item_id,
                                        'date_block_num': prediction_month_batch
                                    }
                                    
                                    # Obtenir les informations du produit
                                    item_info = processed_data[
                                        (processed_data['shop_id'] == shop_id) & 
                                        (processed_data['item_id'] == item_id)
                                    ].iloc[-1]  # Dernière entrée
                                    
                                    prediction_features['item_category_id'] = item_info['item_category_id']
                                    
                                    if 'item_price' in feature_columns:
                                        prediction_features['item_price'] = item_info['item_price']
                                    
                                    # Features de lag
                                    recent_data = processed_data[
                                        (processed_data['shop_id'] == shop_id) & 
                                        (processed_data['item_id'] == item_id)
                                    ].sort_values('date_block_num').tail(3)
                                    
                                    for lag in [1, 2, 3]:
                                        lag_col = f'{sales_col}_lag_{lag}'
                                        if lag_col in feature_columns:
                                            if len(recent_data) >= lag:
                                                prediction_features[lag_col] = recent_data[sales_col].iloc[-lag]
                                            else:
                                                prediction_features[lag_col] = 0
                                    
                                    # Moyenne mobile
                                    ma_col = f'{sales_col}_ma_3'
                                    if ma_col in feature_columns:
                                        if len(recent_data) >= 3:
                                            prediction_features[ma_col] = recent_data[sales_col].tail(3).mean()
                                        else:
                                            prediction_features[ma_col] = recent_data[sales_col].mean() if len(recent_data) > 0 else 0
                                    
                                    # Features agrégées
                                    if any('shop_avg' in col for col in feature_columns):
                                        shop_stats = processed_data[processed_data['shop_id'] == shop_id][sales_col]
                                        prediction_features['shop_avg_sales'] = shop_stats.mean()
                                        prediction_features['shop_std_sales'] = shop_stats.std()
                                    
                                    if any('category_avg' in col for col in feature_columns):
                                        cat_stats = processed_data[processed_data['item_category_id'] == item_info['item_category_id']][sales_col]
                                        prediction_features['category_avg_sales'] = cat_stats.mean()
                                        prediction_features['category_std_sales'] = cat_stats.std()
                                    
                                    prediction_data.append(prediction_features)
                            
                            if prediction_data:
                                # Créer le DataFrame
                                pred_df = pd.DataFrame(prediction_data)
                                
                                # S'assurer que toutes les colonnes sont présentes
                                for col in feature_columns:
                                    if col not in pred_df.columns:
                                        pred_df[col] = 0
                                
                                pred_df = pred_df[feature_columns]
                                
                                # Faire les prédictions
                                predictions = batch_model.predict(pred_df)
                                predictions = np.maximum(predictions, 0)  # Pas de ventes négatives
                                
                                # Créer le DataFrame de résultats
                                results_df = pd.DataFrame(prediction_data)[['shop_id', 'item_id']]
                                results_df['predicted_sales'] = predictions
                                results_df['prediction_month'] = prediction_month_batch
                                results_df['model'] = batch_model_name
                                
                                # Ajouter les noms si disponibles
                                if 'shop_name' in processed_data.columns:
                                    shop_names = processed_data[['shop_id', 'shop_name']].drop_duplicates()
                                    results_df = results_df.merge(shop_names, on='shop_id', how='left')
                                
                                if 'item_name' in processed_data.columns:
                                    item_names = processed_data[['item_id', 'item_name']].drop_duplicates()
                                    results_df = results_df.merge(item_names, on='item_id', how='left')
                                
                                # Trier par prédictions décroissantes
                                results_df = results_df.sort_values('predicted_sales', ascending=False)
                                
                                st.success(f"🎉 {len(results_df)} prédictions générées avec succès !")
                                
                                # Affichage des résultats
                                st.subheader("📊 Résultats des prédictions en lot")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total des ventes prédites", f"{results_df['predicted_sales'].sum():,.0f}")
                                
                                with col2:
                                    st.metric("Ventes moyennes prédites", f"{results_df['predicted_sales'].mean():.1f}")
                                
                                with col3:
                                    st.metric("Prédiction maximale", f"{results_df['predicted_sales'].max():.0f}")
                                
                                # Tableau des résultats
                                st.dataframe(results_df.head(20), use_container_width=True)
                                
                                # Graphiques
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Top 10 des prédictions
                                    top_10 = results_df.head(10)
                                    if 'item_name' in top_10.columns:
                                        fig_top = px.bar(
                                            top_10,
                                            x='predicted_sales',
                                            y='item_name',
                                            orientation='h',
                                            title="Top 10 des prédictions"
                                        )
                                    else:
                                        fig_top = px.bar(
                                            top_10,
                                            x='predicted_sales',
                                            y='item_id',
                                            orientation='h',
                                            title="Top 10 des prédictions"
                                        )
                                    fig_top.update_layout(height=400)
                                    st.plotly_chart(fig_top, use_container_width=True)
                                
                                with col2:
                                    # Distribution des prédictions
                                    fig_dist = px.histogram(
                                        results_df,
                                        x='predicted_sales',
                                        title="Distribution des prédictions",
                                        nbins=30
                                    )
                                    fig_dist.update_layout(height=400)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Sauvegarde des résultats
                                st.session_state['batch_predictions'] = results_df
                                
                                # Bouton de téléchargement
                                csv_buffer = io.StringIO()
                                results_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Télécharger les prédictions (CSV)",
                                    data=csv_data,
                                    file_name=f"predictions_lot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            else:
                                st.warning("Aucune combinaison valide trouvée avec les critères sélectionnés")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération : {str(e)}")
                else:
                    st.error("❌ Veuillez sélectionner au moins un magasin")
        
        with tab3:
            st.subheader("🔮 Prédictions futures")
            st.markdown("Générez des prédictions pour plusieurs mois à venir.")
            
            # Sélection du modèle
            future_model_name = st.selectbox(
                "Modèle pour prédictions futures",
                list(models.keys()),
                key="future_model"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Paramètres temporels:**")
                
                start_month = st.number_input(
                    "Mois de début",
                    min_value=processed_data['date_block_num'].max() + 1,
                    max_value=processed_data['date_block_num'].max() + 24,
                    value=processed_data['date_block_num'].max() + 1,
                    key="future_start"
                )
                
                num_months = st.slider(
                    "Nombre de mois à prédire",
                    min_value=1,
                    max_value=12,
                    value=6
                )
                
                # Sélection d'une combinaison spécifique
                st.markdown("**Combinaison à analyser:**")
                
                # Top combinaisons par ventes historiques
                top_combinations = processed_data.groupby(['shop_id', 'item_id'])[sales_col].sum().sort_values(ascending=False).head(20)
                
                combination_options = {}
                for (shop_id, item_id), sales in top_combinations.items():
                    shop_name = processed_data[processed_data['shop_id'] == shop_id]['shop_name'].iloc[0] if 'shop_name' in processed_data.columns else f"Shop {shop_id}"
                    item_name = processed_data[processed_data['item_id'] == item_id]['item_name'].iloc[0] if 'item_name' in processed_data.columns else f"Item {item_id}"
                    combination_options[f"{shop_name} - {item_name} (Ventes: {sales:.0f})"] = (shop_id, item_id)
                
                selected_combination_display = st.selectbox(
                    "Combinaison magasin-produit",
                    list(combination_options.keys())
                )
                
                selected_shop_id, selected_item_id = combination_options[selected_combination_display]
            
            with col2:
                st.markdown("**Aperçu de la sélection:**")
                
                st.metric("Mois de début", start_month)
                st.metric("Mois de fin", start_month + num_months - 1)
                st.metric("Nombre de prédictions", num_months)
                
                # Données historiques de la combinaison
                historical_data = processed_data[
                    (processed_data['shop_id'] == selected_shop_id) & 
                    (processed_data['item_id'] == selected_item_id)
                ]
                
                if len(historical_data) > 0:
                    st.metric("Ventes historiques moyennes", f"{historical_data[sales_col].mean():.1f}")
                    st.metric("Dernière vente", f"{historical_data[sales_col].iloc[-1]:.0f}")
            
            # Bouton de génération
            if st.button("🚀 Générer les prédictions futures", type="primary"):
                with st.spinner("Génération des prédictions futures..."):
                    try:
                        future_model = models[future_model_name]
                        feature_columns = st.session_state['feature_columns']
                        
                        future_predictions = []
                        
                        # Données de base pour cette combinaison
                        base_data = processed_data[
                            (processed_data['shop_id'] == selected_shop_id) & 
                            (processed_data['item_id'] == selected_item_id)
                        ].sort_values('date_block_num')
                        
                        if len(base_data) == 0:
                            st.error("Aucune donnée historique pour cette combinaison")
                        else:
                            # Utiliser les dernières valeurs comme base
                            last_data = base_data.iloc[-1]
                            
                            # Créer une série temporelle des ventes pour les lags
                            sales_history = base_data[sales_col].tolist()
                            
                            for month_offset in range(num_months):
                                prediction_month = start_month + month_offset
                                
                                # Préparer les features
                                prediction_features = {
                                    'shop_id': selected_shop_id,
                                    'item_id': selected_item_id,
                                    'date_block_num': prediction_month,
                                    'item_category_id': last_data['item_category_id']
                                }
                                
                                if 'item_price' in feature_columns:
                                    prediction_features['item_price'] = last_data['item_price']
                                
                                # Features de lag (utiliser l'historique + prédictions précédentes)
                                current_history = sales_history.copy()
                                
                                for lag in [1, 2, 3]:
                                    lag_col = f'{sales_col}_lag_{lag}'
                                    if lag_col in feature_columns:
                                        if len(current_history) >= lag:
                                            prediction_features[lag_col] = current_history[-lag]
                                        else:
                                            prediction_features[lag_col] = 0
                                
                                # Moyenne mobile
                                ma_col = f'{sales_col}_ma_3'
                                if ma_col in feature_columns:
                                    if len(current_history) >= 3:
                                        prediction_features[ma_col] = np.mean(current_history[-3:])
                                    else:
                                        prediction_features[ma_col] = np.mean(current_history) if current_history else 0
                                
                                # Features agrégées (utiliser les moyennes historiques)
                                if any('shop_avg' in col for col in feature_columns):
                                    shop_stats = processed_data[processed_data['shop_id'] == selected_shop_id][sales_col]
                                    prediction_features['shop_avg_sales'] = shop_stats.mean()
                                    prediction_features['shop_std_sales'] = shop_stats.std()
                                
                                if any('category_avg' in col for col in feature_columns):
                                    cat_stats = processed_data[processed_data['item_category_id'] == last_data['item_category_id']][sales_col]
                                    prediction_features['category_avg_sales'] = cat_stats.mean()
                                    prediction_features['category_std_sales'] = cat_stats.std()
                                
                                # Créer le DataFrame pour prédiction
                                pred_df = pd.DataFrame([prediction_features])
                                
                                for col in feature_columns:
                                    if col not in pred_df.columns:
                                        pred_df[col] = 0
                                
                                pred_df = pred_df[feature_columns]
                                
                                # Faire la prédiction
                                prediction = future_model.predict(pred_df)[0]
                                prediction = max(0, prediction)
                                
                                future_predictions.append({
                                    'month': prediction_month,
                                    'predicted_sales': prediction
                                })
                                
                                # Ajouter la prédiction à l'historique pour les prochaines prédictions
                                current_history.append(prediction)
                                sales_history.append(prediction)
                            
                            # Créer le DataFrame des résultats
                            future_df = pd.DataFrame(future_predictions)
                            
                            st.success(f"🎉 {len(future_df)} prédictions futures générées !")
                            
                            # Affichage des résultats
                            st.subheader("📈 Prédictions futures")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total prédit", f"{future_df['predicted_sales'].sum():.0f}")
                            
                            with col2:
                                st.metric("Moyenne mensuelle", f"{future_df['predicted_sales'].mean():.1f}")
                            
                            with col3:
                                st.metric("Tendance", 
                                    "↗️ Croissante" if future_df['predicted_sales'].iloc[-1] > future_df['predicted_sales'].iloc[0] 
                                    else "↘️ Décroissante" if future_df['predicted_sales'].iloc[-1] < future_df['predicted_sales'].iloc[0]
                                    else "➡️ Stable"
                                )
                            
                            # Graphique des prédictions futures
                            fig_future = go.Figure()
                            
                            # Données historiques
                            if len(base_data) > 0:
                                fig_future.add_trace(go.Scatter(
                                    x=base_data['date_block_num'],
                                    y=base_data[sales_col],
                                    mode='lines+markers',
                                    name='Historique',
                                    line=dict(color='blue')
                                ))
                            
                            # Prédictions futures
                            fig_future.add_trace(go.Scatter(
                                x=future_df['month'],
                                y=future_df['predicted_sales'],
                                mode='lines+markers',
                                name='Prédictions',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig_future.update_layout(
                                title="Évolution historique et prédictions futures",
                                xaxis_title="Mois",
                                yaxis_title="Ventes",
                                height=500
                            )
                            
                            st.plotly_chart(fig_future, use_container_width=True)
                            
                            # Tableau des prédictions
                            st.subheader("📋 Détail des prédictions")
                            st.dataframe(future_df, use_container_width=True)
                            
                            # Sauvegarde
                            st.session_state['future_predictions'] = future_df
                            
                            # Téléchargement
                            csv_buffer = io.StringIO()
                            future_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Télécharger les prédictions futures (CSV)",
                                data=csv_data,
                                file_name=f"predictions_futures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération : {str(e)}")
        
        # Historique des prédictions
        if 'custom_predictions' in st.session_state and st.session_state['custom_predictions']:
            st.subheader("📚 Historique des prédictions personnalisées")
            
            predictions_history = pd.DataFrame(st.session_state['custom_predictions'])
            predictions_history['timestamp'] = pd.to_datetime(predictions_history['timestamp'])
            predictions_history = predictions_history.sort_values('timestamp', ascending=False)
            
            st.dataframe(predictions_history, use_container_width=True)
            
            # Bouton pour effacer l'historique
            if st.button("🗑️ Effacer l'historique"):
                st.session_state['custom_predictions'] = []
                st.rerun()

# Point d'entrée de l'application
if __name__ == "__main__":
    app = SalesPredictionApp()
    app.run()
