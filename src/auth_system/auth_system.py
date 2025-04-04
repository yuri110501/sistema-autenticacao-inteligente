import numpy as np
import torch
from typing import Dict, List, Optional, Any
import random

from src.models.network import AuthenticationAgent
from src.utils.aco import ACOOptimizer

class AuthenticationSystem:
    """
    Sistema de autenticação inteligente que utiliza diferentes métodos
    de autenticação para diferentes setores de uma empresa.
    """
    def __init__(self):
        self.sectors = {
            "Financeiro": "biometria",
            "Servidores": "senha",
            "Laboratorio": "token",
            "Diretoria": "reconhecimento_facial"
        }
        
        self.users = {
            "user1": {"setor": "Financeiro", "biometria": [0.8, 0.9, 0.7]},
            "user2": {"setor": "Servidores", "senha": "senha123"},
            "user3": {"setor": "Laboratorio", "token": "token456"},
            "user4": {"setor": "Diretoria", "reconhecimento_facial": [0.9, 0.8, 0.9]}
        }
        
        self.agent = AuthenticationAgent(input_size=3, hidden_size=64, output_size=4)
        self.aco = ACOOptimizer(n_ants=10, n_iterations=100)
        self.aco.initialize_pheromone(len(self.users))
        
    def predict_sector(self, input_data: torch.Tensor) -> str:
        """
        Prevê o setor com base nos dados de entrada
        
        Args:
            input_data: Tensor de entrada para o modelo neural
            
        Returns:
            Nome do setor previsto
        """
        with torch.no_grad():
            output = self.agent(input_data)
            sector_idx = torch.argmax(output).item()
            return list(self.sectors.keys())[sector_idx]
            
    def validate_authentication(self, user_id: str, auth_type: str, auth_data: Any) -> bool:
        """
        Valida a autenticação para um usuário
        
        Args:
            user_id: ID do usuário
            auth_type: Tipo de autenticação (biometria, senha, token, reconhecimento_facial)
            auth_data: Dados de autenticação
            
        Returns:
            True se a autenticação for válida, False caso contrário
        """
        user = self.users.get(user_id)
        if not user:
            return False
            
        required_auth = self.sectors[user["setor"]]
        if auth_type != required_auth:
            return False
            
        if auth_type == "biometria":
            return self._validate_biometry(user[auth_type], auth_data)
        elif auth_type == "senha":
            return user[auth_type] == auth_data
        elif auth_type == "token":
            return user[auth_type] == auth_data
        elif auth_type == "reconhecimento_facial":
            return self._validate_facial_recognition(user[auth_type], auth_data)
            
        return False
        
    def _validate_biometry(self, stored: List[float], provided: List[float]) -> bool:
        """Valida dados biométricos"""
        return np.mean(np.abs(np.array(stored) - np.array(provided))) < 0.2
        
    def _validate_facial_recognition(self, stored: List[float], provided: List[float]) -> bool:
        """Valida reconhecimento facial"""
        return np.mean(np.abs(np.array(stored) - np.array(provided))) < 0.15
        
    def find_user(self, auth_data: Any) -> Optional[str]:
        """
        Busca um usuário com base nos dados de autenticação
        
        Args:
            auth_data: Dados de autenticação
            
        Returns:
            ID do usuário encontrado ou None se nenhum usuário for encontrado
        """
        best_score = 0.0
        best_user = None
        
        for user_id, user_data in self.users.items():
            score = self._calculate_similarity(user_data, auth_data)
            if score > best_score:
                best_score = score
                best_user = user_id
                
        return best_user if best_score > 0.5 else None
        
    def _calculate_similarity(self, user_data: Dict, auth_data: Any) -> float:
        """
        Calcula a similaridade entre os dados do usuário e os dados de autenticação
        
        Args:
            user_data: Dados do usuário
            auth_data: Dados de autenticação
            
        Returns:
            Score de similaridade entre 0 e 1
        """
        # 1. Se for dados biométricos (lista de floats)
        if isinstance(auth_data, list) and all(isinstance(x, (int, float)) for x in auth_data):
            # Verifica biometria
            if "biometria" in user_data and isinstance(user_data["biometria"], list):
                similarity = 1 - min(1.0, np.mean(np.abs(np.array(user_data["biometria"]) - np.array(auth_data))))
                return similarity
                
            # Verifica reconhecimento facial
            if "reconhecimento_facial" in user_data and isinstance(user_data["reconhecimento_facial"], list):
                similarity = 1 - min(1.0, np.mean(np.abs(np.array(user_data["reconhecimento_facial"]) - np.array(auth_data))))
                return similarity
        
        # 2. Se for string (senha ou token)
        elif isinstance(auth_data, str):
            # Verifica senha
            if "senha" in user_data and user_data["senha"] == auth_data:
                return 1.0
                
            # Verifica token
            if "token" in user_data and user_data["token"] == auth_data:
                return 1.0
                
        # Nenhuma correspondência encontrada
        return 0.0 