#!/usr/bin/env python3
"""
Script para executar a API do Sistema de Autenticação Inteligente
"""
from src.api.fastapi_app import start_server

if __name__ == "__main__":
    print("Iniciando API do Sistema de Autenticação Inteligente...")
    print("Acesse a API em http://localhost:8000")
    print("Para acessar a documentação, acesse http://localhost:8000/docs")
    start_server(host="0.0.0.0", port=8000) 