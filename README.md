# Sistema de Autenticação Inteligente

Sistema de autenticação baseado em IA que utiliza diferentes métodos de autenticação para diferentes setores de uma empresa.

## Funcionalidades

- Agente inteligente único para múltiplos setores
- Diferentes métodos de autenticação por setor:
  - Financeiro: Biometria
  - Servidores: Senha
  - Laboratório: Token
  - Diretoria: Reconhecimento facial
- Algoritmo de busca baseado em colônia de formigas (ACO)
- Few-shot learning para lidar com poucos dados
- Interface web para testes
- API REST para integração

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Execução

1. Inicie a API:
```bash
python api.py
```

2. Em outro terminal, inicie a interface web:
```bash
streamlit run app.py
```

## Estrutura do Projeto

- `agent.py`: Implementação do agente de autenticação
- `api.py`: API FastAPI
- `app.py`: Interface Streamlit
- `requirements.txt`: Dependências do projeto

## Exemplos de Uso

### Via API

```python
import requests

response = requests.post(
    "http://localhost:8000/authenticate",
    json={
        "auth_type": "biometria",
        "data": [0.8, 0.9, 0.7]
    }
)
print(response.json())
```

### Via Interface Web

1. Acesse `http://localhost:8501`
2. Selecione o tipo de autenticação
3. Insira os dados de autenticação
4. Clique em "Autenticar"

## Melhorias de Performance

O sistema inclui várias técnicas para otimização de performance:

1. **Pruning**: Redução do tamanho do modelo neural
2. **Quantização**: Conversão para tipos de dados menores
3. **ONNX**: Exportação para formato otimizado
4. **Runtime leve**: Implementação eficiente do ACO

## Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças propostas.

## Licença

MIT 