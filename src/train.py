import os
import torch
import numpy as np
import galsim
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos tu pipeline de datos y tu red neuronal de doble etapa
from dataset import OnlineAstronomyDataset
from model import UNet48


#Primero debemos hacer un split de los datos
#Para ellos, se permutan los indices de las galaxias, y se asignan 5000 para entrenamiento y 1000 para validación
def index_split(total_galaxies, n_train, n_val, seed=42):
    assert n_train + n_val <= total_galaxies, "No hay suficientes galaxias para el split."
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_galaxies)#Mezcla aleatoria de los indices de las galaxias

    #Se asignan indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    return train_idx, val_idx


def train_model():
    # 1. Configuración del Dispositivo (GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando pipeline. Dispositivo de entrenamiento: {device}")

    # 2. Hiperparámetros de entrenamiento
    BATCH_SIZE = 40     # Tamaño del lote 
    LEARNING_RATE = 1e-3   # Tasa de aprendizaje inicial estándar (0.001) este learning rate
    EPOCHS = 100            # Número de épocas para entrenar
    N_TRAIN= 5000
    N_VAL= 1000
    #Datos del catálogo COSMOS
    CATALOG_FILE  = "real_galaxy_catalog_23.5.fits"
    CATALOG_DIR   = "/home/luifer/BlindDeconvolutionAstronomy/data/COSMOS_23.5_training_sample"
    #Obtenemos el maximo de galaxias del catalogo COSMOS
    TOTAL_GALAXIES = galsim.COSMOSCatalog(file_name=CATALOG_FILE, dir=CATALOG_DIR).nobjects


    # 3. Inicializar el Dataset y el DataLoader de PyTorch
    train_idx, val_idx = index_split(TOTAL_GALAXIES, N_TRAIN, N_VAL) #Hacemos el split de indices para entrenamiento y validación
    train_dataset= OnlineAstronomyDataset(index_pool=train_idx, catalog_file=CATALOG_FILE, catalog_dir=CATALOG_DIR) #Dataset de entrenamiento con los indices asignados
    val_dataset= OnlineAstronomyDataset(index_pool=val_idx, catalog_file=CATALOG_FILE, catalog_dir=CATALOG_DIR) #Dataset de validación con los indices asignados


    # El DataLoader baraja las galaxias (shuffle=True) y las agrupa en lotes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Instanciar el modelo de Dos Etapas y enviarlo a la GPU
    model = UNet48(in_channels=1, out_channels=1).to(device)

    # 5. Definir la Función de Pérdida (Loss) y el Optimizador
    # MSE calcula la diferencia pixel por pixel entre la predicción de la IA y el Hubble real
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #cargar pesos previos
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("▶️  Pesos cargados, arrancando desde época 0")

    print(f"🏋️ En sus marcas... ¡A entrenar! Total de épocas: {EPOCHS}")
    print("-" * 50)
    best_val_loss = float('inf')  # Para guardar el mejor modelo basado en la pérdida de validación

    # 6. Bucle Principal de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()  # Activa el modo de entrenamiento (activa BatchNorm)
        train_loss = 0.0

        #Proceso de entrenamiento
        for batch_x_o, batch_x_t in train_loader:    
            # Enviar las imágenes al dispositivo (GPU)
            batch_x_o = batch_x_o.to(device) # Input sucio
            batch_x_t = batch_x_t.to(device) # Target limpio

            # --- El ciclo clásico de PyTorch ---
            optimizer.zero_grad()               # 1. Limpiar los gradientes del paso anterior
            outputs = model(batch_x_o)          # 2. Forward pass: la UNet intenta limpiar la imagen
            loss = criterion(outputs, batch_x_t) # 3. Calcular el error contra el Target real
            loss.backward()                     # 4. Backward pass: calcular cómo ajustar los pesos
            optimizer.step()                    # 5. Optimizar: aplicar los ajustes matemáticos

            train_loss = loss.item()

        #Validación al final de cada época
        model.eval() # Modo evaluación (desactiva BatchNorm)
        val_loss = 0.0
        with torch.no_grad(): # No necesitamos calcular gradientes para la validación
            for batch_x_o, batch_x_t in val_loader:
                batch_x_o = batch_x_o.to(device)
                batch_x_t = batch_x_t.to(device)

                outputs = model(batch_x_o)
                loss = criterion(outputs, batch_x_t)
                val_loss += loss.item()
        #Se promedio la perdida de val_loss
        val_loss/= len(val_loader)
        # Mostrar la pérdida tras cada época
        print("-" * 50)
        print(f"✅ ÉPOCA {epoch+1} TERMINADA | Pérdida Promedio de train: {train_loss:.5f}")
        print(f"✅ ÉPOCA {epoch+1} TERMINADA | Pérdida Promedio de validation: {val_loss:.5f}")
        print("-" * 50)

        #7. Guardar el modelo al final de cada época solo si val_loss < best_val_loss
        checkpoint_path = "best_model.pth"
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Época [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val Loss: {best_val_loss:.5f} ✅ mejor modelo guardado")
            print(f"💾 Modelo guardado exitosamente en: {checkpoint_path}\n")
        
    print("🏁 ¡Entrenamiento completado con éxito!")

if __name__ == "__main__":
    train_model()