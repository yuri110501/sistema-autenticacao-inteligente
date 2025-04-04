#!/usr/bin/env python3
"""
Script para executar a interface web do Sistema de Autenticação Inteligente
"""
import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    print("Iniciando interface web do Sistema de Autenticação Inteligente...")
    print("Acesse a interface em http://localhost:8501")
    sys.argv = ["streamlit", "run", os.path.join("src", "api", "streamlit_app.py")]
    sys.exit(stcli.main()) 