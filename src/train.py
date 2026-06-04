import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos tu pipeline de datos y tu red neuronal de doble etapa
from src.dataset import OnlineAstronomyDataset
from src.model import DualStageUNet48

def train_model():
    # 1. Configuración del Dispositivo (Automático para tu CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando pipeline. Dispositivo de entrenamiento: {device}")

    # 2. Hiperparámetros de entrenamiento
    BATCH_SIZE = 16        # Tamaño del lote (ligero y seguro para la memoria RAM)
    LEARNING_RATE = 1e-3   # Tasa de aprendizaje inicial estándar (0.001)
    EPOCHS = 5             # Empezamos con 5 épocas para medir los tiempos en tu CPU
    STEPS_PER_EPOCH = 100  # ¡CLAVE! Limitamos a 100 pasos por época para que no tarde horas en CPU

    # 3. Inicializar el Dataset y el DataLoader de PyTorch
    print("📦 Cargando catálogo COSMOS del Hubble...")
    dataset = OnlineAstronomyDataset(pixel_scale=0.03)
    
    # El DataLoader baraja las galaxias (shuffle=True) y las agrupa en lotes
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. Instanciar el modelo de Dos Etapas y enviarlo a la CPU
    model = DualStageUNet48(in_channels=1, out_channels=1).to(device)

    # 5. Definir la Función de Pérdida (Loss) y el Optimizador
    # MSE calcula la diferencia pixel por pixel entre la predicción de la IA y el Hubble real
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🏋️ En sus marcas... ¡A entrenar! Total de épocas: {EPOCHS}")
    print("-" * 50)

    # 6. Bucle Principal de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()  # Activa el modo de entrenamiento (activa BatchNorm)
        running_loss = 0.0
        
        for step, (batch_x_o, batch_x_t) in enumerate(train_loader):
            # Si alcanzamos el límite de pasos por época, cortamos (estrategia para CPU)
            if step >= STEPS_PER_EPOCH:
                break
                
            # Enviar las imágenes al dispositivo (CPU)
            batch_x_o = batch_x_o.to(device) # Input sucio
            batch_x_t = batch_x_t.to(device) # Target limpio

            # --- El ciclo clásico de PyTorch ---
            optimizer.zero_grad()               # 1. Limpiar los gradientes del paso anterior
            outputs = model(batch_x_o)          # 2. Forward pass: la IA intenta limpiar la imagen
            loss = criterion(outputs, batch_x_t) # 3. Calcular el error contra el Target real
            loss.backward()                     # 4. Backward pass: calcular cómo ajustar los pesos
            optimizer.step()                    # 5. Optimizar: aplicar los ajustes matemáticos

            running_loss += loss.item()

            # Imprimir el progreso cada 20 lotes procesados
            if (step + 1) % 20 == 0:
                print(f"Época [{epoch+1}/{EPOCHS}] | Paso [{step+1}/{STEPS_PER_EPOCH}] | Pérdida (MSE): {loss.item():.5f}")

        # Calcular y mostrar la pérdida promedio de la época
        epoch_loss = running_loss / STEPS_PER_EPOCH
        print("-" * 50)
        print(f"✅ ÉPOCA {epoch+1} TERMINADA | Pérdida Promedio: {epoch_loss:.5f}")
        print("-" * 50)

        # 7. Guardar el modelo al final de cada época
        checkpoint_path = "best_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Modelo guardado exitosamente en: {checkpoint_path}\n")

    print("🏁 ¡Entrenamiento completado con éxito!")

if __name__ == "__main__":
    train_model()