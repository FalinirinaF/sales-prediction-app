"""
Utilitaires pour les modèles de machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, Tuple, Any

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features pour l'entraînement du modèle
    
    Args:
        df: DataFrame avec les données mensuelles
        
    Returns:
        DataFrame avec les features préparées
    """
    df_features = df.copy()
    
    # Features de base
    features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_price']
    
    # Ajout de features dérivées
    # Lag features (ventes des mois précédents)
    df_features = df_features.sort_values(['shop_id', 'item_id', 'date_block_num'])
    
    for lag in [1, 2, 3]:
        df_features[f'item_cnt_month_lag_{lag}'] = df_features.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
    
    # Moyennes mobiles
    df_features['item_cnt_month_ma_3'] = df_features.groupby(['shop_id', 'item_id'])['item_cnt_month'].rolling(3).mean().reset_index(0, drop=True)
    
    # Features agrégées par magasin et catégorie
    shop_stats = df_features.groupby(['shop_id', 'date_block_num'])['item_cnt_month'].agg(['mean', 'std']).reset_index()
    shop_stats.columns = ['shop_id', 'date_block_num', 'shop_avg_sales', 'shop_std_sales']
    
    category_stats = df_features.groupby(['item_category_id', 'date_block_num'])['item_cnt_month'].agg(['mean', 'std']).reset_index()
    category_stats.columns = ['item_category_id', 'date_block_num', 'category_avg_sales', 'category_std_sales']
    
    # Merge des statistiques
    df_features = df_features.merge(shop_stats, on=['shop_id', 'date_block_num'], how='left')
    df_features = df_features.merge(category_stats, on=['item_category_id', 'date_block_num'], how='left')
    
    return df_features

def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """
    Entraîne plusieurs modèles et retourne leurs performances
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        
    Returns:
        Dict contenant les modèles entraînés et leurs métriques
    """
    models = {}
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'mae': mean_absolute_error(y_val, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, lr_pred)),
        'r2': r2_score(y_val, lr_pred)
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'mae': mean_absolute_error(y_val, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, rf_pred)),
        'r2': r2_score(y_val, rf_pred)
    }
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'mae': mean_absolute_error(y_val, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred)),
        'r2': r2_score(y_val, xgb_pred)
    }
    
    return {'models': models, 'results': results}

def get_feature_importance(model, feature_names: list, model_type: str) -> pd.DataFrame:
    """
    Extrait l'importance des features selon le type de modèle
    
    Args:
        model: Modèle entraîné
        feature_names: Liste des noms de features
        model_type: Type de modèle
        
    Returns:
        DataFrame avec l'importance des features
    """
    if model_type == 'Linear Regression':
        importance = np.abs(model.coef_)
    elif model_type in ['Random Forest', 'XGBoost']:
        importance = model.feature_importances_
    else:
        return pd.DataFrame()
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance
