from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
from agent import AuthenticationSystem

app = FastAPI(title="Sistema de Autenticação Inteligente")
system = AuthenticationSystem()

class AuthRequest(BaseModel):
    auth_type: str
    data: Union[List[float], str]

class AuthResponse(BaseModel):
    user_id: Optional[str]
    sector: Optional[str]
    is_valid: bool
    message: str

@app.post("/authenticate", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    try:
        user_id = system.find_user(request.data)
        if not user_id:
            return AuthResponse(
                user_id=None,
                sector=None,
                is_valid=False,
                message="Usuário não encontrado"
            )
            
        sector = system.users[user_id]["setor"]
        is_valid = system.validate_authentication(
            user_id,
            request.auth_type,
            request.data
        )
        
        return AuthResponse(
            user_id=user_id,
            sector=sector,
            is_valid=is_valid,
            message="Autenticação realizada com sucesso" if is_valid else "Autenticação falhou"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors")
async def get_sectors():
    return system.sectors

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 