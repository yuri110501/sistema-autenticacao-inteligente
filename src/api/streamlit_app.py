import streamlit as st
import requests
import json
from typing import List, Union

def run_app(api_url: str = "http://localhost:8000"):
    """
    Executa a aplicação Streamlit
    
    Args:
        api_url: URL da API
    """
    st.title("Sistema de Autenticação Inteligente")
    st.write("Teste o sistema de autenticação com diferentes métodos")

    # Obter setores disponíveis
    try:
        response = requests.get(f"{api_url}/sectors")
        sectors = response.json()
    except:
        st.error("Não foi possível conectar à API. Certifique-se de que o servidor está rodando.")
        st.stop()

    # Interface de autenticação
    auth_type = st.selectbox(
        "Tipo de Autenticação",
        list(sectors.values())
    )

    if auth_type in ["biometria", "reconhecimento_facial"]:
        data = st.text_input(
            "Dados de Autenticação (separados por vírgula)",
            help="Exemplo: 0.8,0.9,0.7"
        )
        try:
            auth_data = [float(x.strip()) for x in data.split(",") if x.strip()]
            if not auth_data:
                st.error("Por favor, insira os dados de autenticação.")
                st.stop()
        except:
            st.error("Formato inválido. Use números separados por vírgula.")
            st.stop()
    else:
        auth_data = st.text_input("Dados de Autenticação")

    if st.button("Autenticar"):
        try:
            response = requests.post(
                f"{api_url}/authenticate",
                json={
                    "auth_type": auth_type,
                    "data": auth_data
                }
            )
            result = response.json()
            
            # Verifica se a resposta contém user_id e se ele não é None
            if result.get("user_id"):
                st.success(f"Usuário encontrado: {result['user_id']}")
                st.info(f"Setor: {result['sector']}")
                if result["is_valid"]:
                    st.success("Autenticação válida!")
                else:
                    st.error("Autenticação inválida!")
            else:
                st.warning("Usuário não encontrado")
                st.info(f"Mensagem: {result.get('message', 'Nenhuma mensagem disponível')}")
                
        except Exception as e:
            st.error(f"Erro na autenticação: {str(e)}")
            st.info("Detalhes: Verifique se o formato dos dados está correto para o tipo de autenticação")

    # Exemplos de uso
    st.sidebar.title("Exemplos de Uso")
    st.sidebar.write("Use estes exemplos para testar o sistema:")

    examples = {
        "Biometria": "0.8,0.9,0.7",
        "Senha": "senha123",
        "Token": "token456",
        "Reconhecimento Facial": "0.9,0.8,0.9"
    }

    for auth, data in examples.items():
        st.sidebar.write(f"**{auth}**:")
        st.sidebar.code(data)

if __name__ == "__main__":
    run_app() 