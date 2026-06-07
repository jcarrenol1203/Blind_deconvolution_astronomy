import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos tu pipeline de datos y tu red neuronal de doble etapa
from src.dataset import OnlineAstronomyDataset
from src.model import DualStageUNet48

def train_model():
    # 1. Configuración del Dispositivo (Automático para CPU o GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando pipeline. Dispositivo de entrenamiento: {device}")

    # 2. Configuración Inteligente y Multi-usuario de Rutas para el Catálogo
    POSIBLES_RUTAS = [
        "/home/luifer/BlindDeconvolutionAstronomy/data/COSMOS_23.5_training_sample",
        "/home/jacl/github/Blind_deconvolution_astronomy/data/COSMOS_23.5_training_sample",
        "./data/COSMOS_23.5_training_sample"  # Ruta relativa de respaldo
    ]

    catalog_dir = None
    for ruta in POSIBLES_RUTAS:
        if os.path.exists(ruta):
            catalog_dir = ruta
            break

    if catalog_dir is not None:
        print(f"📂 Carpeta de datos detectada y mapeada con éxito en: {catalog_dir}")
    else:
        # Si no se encuentra en disco, dejamos que GalSim use su descarga interna por defecto
        print("ℹ️ No se detectó ninguna carpeta local explícita. GalSim usará la ruta del entorno virtual.")

    # 3. Hiperparámetros de entrenamiento
    BATCH_SIZE = 8         # Tamaño del lote (ligero y seguro para la memoria RAM)
    LEARNING_RATE = 1e-3   # Tasa de aprendizaje inicial estándar (0.001)
    EPOCHS = 15            # Cantidad total de épocas a entrenar
    STEPS_PER_EPOCH = 100  # Limitamos a 100 pasos por época para control de tiempos en CPU

    # 4. Inicializar el Dataset y el DataLoader de PyTorch
    print("📦 Cargando catálogo COSMOS del Hubble...")
    # Pasamos la carpeta detectada dinámicamente al inicializador del dataset
    dataset = OnlineAstronomyDataset(catalog_dir=catalog_dir, pixel_scale=0.03)
    
    # El DataLoader baraja las galaxias (shuffle=True) y las agrupa en lotes
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 5. Instanciar el modelo de Dos Etapas y enviarlo al dispositivo
    model = DualStageUNet48(in_channels=1, out_channels=1).to(device)

    # 6. Definir la Función de Pérdida (Loss) y el Optimizador
    # MSE calcula la diferencia pixel por pixel entre la predicción de la IA y el Hubble real
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Variables de control para guardar la mejor época (menor error acumulado)
    best_val_loss = float('inf') 
    checkpoint_path = "best_model.pth"

    print(f"🏋️ En sus marcas... ¡A entrenar! Total de épocas: {EPOCHS}")
    print("-" * 65)

    # 7. Bucle Principal de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()  # Activa el modo de entrenamiento (activa BatchNorm)
        running_loss = 0.0
        
        for step, (batch_x_o, batch_x_t) in enumerate(train_loader):
            # Si alcanzamos el límite de pasos por época, cortamos (estrategia para CPU)
            if step >= STEPS_PER_EPOCH:
                break
                
            # Enviar las imágenes al dispositivo (CPU/GPU)
            batch_x_o = batch_x_o.to(device) # Input sucio (imagen observada degradada)
            batch_x_t = batch_x_t.to(device) # Target limpio (ground truth)

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

        # Calcular la pérdida promedio de la época actual
        epoch_loss = running_loss / STEPS_PER_EPOCH
        print("-" * 65)
        print(f"✅ ÉPOCA {epoch+1} TERMINADA | Pérdida Promedio: {epoch_loss:.5f}")
        
        # 8. Guardar el modelo ÚNICAMENTE si la pérdida actual mejora el récord histórico
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🔥 ¡Nueva mejor pérdida conseguida! ({best_val_loss:.5f}) ➔ Guardando pesos en '{checkpoint_path}'...")
        else:
            print(f"💤 No hubo mejora en esta época. El mejor error sigue siendo: {best_val_loss:.5f}")
            
        print("-" * 65 + "\n")

    print(f"🏁 ¡Entrenamiento completado con éxito! Mejor pérdida final conservada: {best_val_loss:.5f}")

if __name__ == "__main__":
    train_model()