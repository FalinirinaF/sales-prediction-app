"""
Utilitaires pour le traitement des données de ventes
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def validate_data_format(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Valide le format des données importées
    
    Args:
        df: DataFrame à valider
        
    Returns:
        Dict contenant les résultats de validation
    """
    required_columns = [
        'shop_id', 'item_id', 'item_category_id', 'item_cnt_day',
        'item_price', 'date', 'date_block_num', 'item_name',
        'shop_name', 'item_category_name'
    ]
    
    validation_results = {
        'has_required_columns': all(col in df.columns for col in required_columns),
        'has_data': len(df) > 0,
        'date_format_valid': True,
        'numeric_columns_valid': True
    }
    
    # Vérification du format des dates
    try:
        pd.to_datetime(df['date'], format='%d/%m/%Y', errors='raise')
    except:
        validation_results['date_format_valid'] = False
    
    # Vérification des colonnes numériques
    numeric_columns = ['shop_id', 'item_id', 'item_category_id', 'item_cnt_day', 'item_price', 'date_block_num']
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_results['numeric_columns_valid'] = False
                break
    
    return validation_results

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Génère un résumé des données
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict contenant les statistiques du dataset
    """
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'unique_shops': df['shop_id'].nunique() if 'shop_id' in df.columns else 0,
        'unique_items': df['item_id'].nunique() if 'item_id' in df.columns else 0,
        'unique_categories': df['item_category_id'].nunique() if 'item_category_id' in df.columns else 0,
        'date_range': {
            'min_date_block': df['date_block_num'].min() if 'date_block_num' in df.columns else None,
            'max_date_block': df['date_block_num'].max() if 'date_block_num' in df.columns else None
        }
    }

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données (supprime doublons, traite valeurs manquantes)
    
    Args:
        df: DataFrame à nettoyer
        
    Returns:
        DataFrame nettoyé
    """
    df_clean = df.copy()
    
    # Suppression des doublons
    df_clean = df_clean.drop_duplicates()
    
    # Traitement des valeurs manquantes
    # Pour les prix, on peut utiliser la médiane par catégorie
    if 'item_price' in df_clean.columns and 'item_category_id' in df_clean.columns:
        df_clean['item_price'] = df_clean.groupby('item_category_id')['item_price'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Pour les quantités vendues, on remplace par 0
    if 'item_cnt_day' in df_clean.columns:
        df_clean['item_cnt_day'] = df_clean['item_cnt_day'].fillna(0)
    
    return df_clean

def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les ventes quotidiennes en ventes mensuelles
    
    Args:
        df: DataFrame avec ventes quotidiennes
        
    Returns:
        DataFrame avec ventes mensuelles agrégées
    """
    # Grouper par shop_id, item_id et date_block_num pour obtenir les ventes mensuelles
    monthly_sales = df.groupby(['shop_id', 'item_id', 'date_block_num']).agg({
        'item_cnt_day': 'sum',  # Somme des ventes quotidiennes = ventes mensuelles
        'item_price': 'mean',   # Prix moyen du mois
        'item_category_id': 'first',
        'item_name': 'first',
        'shop_name': 'first',
        'item_category_name': 'first'
    }).reset_index()
    
    # Renommer la colonne des ventes
    monthly_sales = monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'})
    
    return monthly_sales
