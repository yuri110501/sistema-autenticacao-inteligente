import numpy as np
from typing import List

class ACOOptimizer:
    """
    Implementação do algoritmo de colônia de formigas (Ant Colony Optimization)
    para busca de usuários no sistema de autenticação.
    """
    def __init__(self, n_ants: int, n_iterations: int, alpha: float = 1.0, beta: float = 2.0):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Peso do feromônio
        self.beta = beta    # Peso da heurística
        self.pheromone = None
        
    def initialize_pheromone(self, n_users: int):
        """Inicializa a matriz de feromônios"""
        self.pheromone = np.ones((n_users, n_users)) * 0.1
        
    def update_pheromone(self, paths: List[List[int]], scores: List[float]):
        """Atualiza os feromônios com base nos caminhos percorridos e scores"""
        evaporation = 0.1
        self.pheromone *= (1 - evaporation)
        
        for path, score in zip(paths, scores):
            for i in range(len(path)-1):
                self.pheromone[path[i]][path[i+1]] += score
                
    def search(self, distances: np.ndarray) -> List[int]:
        """
        Realiza a busca usando o algoritmo de colônia de formigas
        
        Args:
            distances: Matriz de distâncias entre os nós
            
        Returns:
            Melhor caminho encontrado
        """
        if self.pheromone is None:
            self.initialize_pheromone(distances.shape[0])
            
        best_path = None
        best_score = float('inf')
        
        for _ in range(self.n_iterations):
            paths = []
            scores = []
            
            for _ in range(self.n_ants):
                path = self._construct_path(distances)
                score = self._calculate_path_score(path, distances)
                paths.append(path)
                scores.append(1.0 / score if score > 0 else 1.0)
                
                if score < best_score:
                    best_score = score
                    best_path = path.copy()
                    
            self.update_pheromone(paths, scores)
            
        return best_path
    
    def _construct_path(self, distances: np.ndarray) -> List[int]:
        """Constrói um caminho para uma formiga"""
        n = distances.shape[0]
        visited = [False] * n
        start = np.random.randint(0, n)
        path = [start]
        visited[start] = True
        
        while len(path) < n:
            current = path[-1]
            next_node = self._select_next_node(current, visited, distances)
            path.append(next_node)
            visited[next_node] = True
            
        return path
    
    def _select_next_node(self, current: int, visited: List[bool], distances: np.ndarray) -> int:
        """Seleciona o próximo nó com base em feromônios e distâncias"""
        n = len(visited)
        probabilities = np.zeros(n)
        
        for i in range(n):
            if not visited[i]:
                # Calcula a probabilidade com base em feromônios e distância inversa
                tau = self.pheromone[current][i] ** self.alpha
                eta = (1.0 / (distances[current][i] + 1e-10)) ** self.beta
                probabilities[i] = tau * eta
                
        # Normaliza as probabilidades
        total = np.sum(probabilities)
        if total > 0:
            probabilities = probabilities / total
            
        # Seleciona um nó com base nas probabilidades
        if np.sum(probabilities) > 0:
            return np.random.choice(n, p=probabilities)
        else:
            # Se todas as probabilidades forem zero, escolhe aleatoriamente entre os não visitados
            candidates = [i for i in range(n) if not visited[i]]
            return np.random.choice(candidates) if candidates else 0
    
    def _calculate_path_score(self, path: List[int], distances: np.ndarray) -> float:
        """Calcula o score (comprimento) de um caminho"""
        score = 0.0
        for i in range(len(path) - 1):
            score += distances[path[i]][path[i + 1]]
        # Adiciona a distância de volta ao início (para formar um ciclo)
        score += distances[path[-1]][path[0]]
        return score 