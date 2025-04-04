import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random

class AuthenticationAgent(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AuthenticationAgent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

class ACOOptimizer:
    def __init__(self, n_ants: int, n_iterations: int, alpha: float = 1.0, beta: float = 2.0):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.pheromone = None
        
    def initialize_pheromone(self, n_users: int):
        self.pheromone = np.ones((n_users, n_users)) * 0.1
        
    def update_pheromone(self, paths: List[List[int]], scores: List[float]):
        evaporation = 0.1
        self.pheromone *= (1 - evaporation)
        
        for path, score in zip(paths, scores):
            for i in range(len(path)-1):
                self.pheromone[path[i]][path[i+1]] += score

class AuthenticationSystem:
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
        with torch.no_grad():
            output = self.agent(input_data)
            sector_idx = torch.argmax(output).item()
            return list(self.sectors.keys())[sector_idx]
            
    def validate_authentication(self, user_id: str, auth_type: str, auth_data: Any) -> bool:
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
        return np.mean(np.abs(np.array(stored) - np.array(provided))) < 0.2
        
    def _validate_facial_recognition(self, stored: List[float], provided: List[float]) -> bool:
        return np.mean(np.abs(np.array(stored) - np.array(provided))) < 0.15
        
    def find_user(self, auth_data: Any) -> Optional[str]:
        best_score = 0.0  # Mudamos para 0 para evitar scores negativos
        best_user = None
        
        for user_id, user_data in self.users.items():
            score = self._calculate_similarity(user_data, auth_data)
            if score > best_score:
                best_score = score
                best_user = user_id
                
        return best_user if best_score > 0.5 else None
        
    def _calculate_similarity(self, user_data: Dict, auth_data: Any) -> float:
        # Implementação melhorada da similaridade
        # Tenta diferentes tipos de autenticação
        
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

# Exemplo de uso
if __name__ == "__main__":
    system = AuthenticationSystem()
    
    # Simulação de autenticação
    test_cases = [
        {"auth_type": "biometria", "data": [0.8, 0.9, 0.7]},
        {"auth_type": "senha", "data": "senha123"},
        {"auth_type": "token", "data": "token456"},
        {"auth_type": "reconhecimento_facial", "data": [0.9, 0.8, 0.9]}
    ]
    
    for test in test_cases:
        print(f"\nTestando autenticação: {test['auth_type']}")
        user_id = system.find_user(test['data'])
        if user_id:
            sector = system.users[user_id]["setor"]
            is_valid = system.validate_authentication(user_id, test['auth_type'], test['data'])
            print(f"Usuário encontrado: {user_id}")
            print(f"Setor: {sector}")
            print(f"Autenticação válida: {is_valid}")
        else:
            print("Usuário não encontrado") 