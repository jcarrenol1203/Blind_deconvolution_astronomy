import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Bloque básico de convolución doble (Conv + BatchNorm + ReLU) x 2"""
    def __init__(self, in_channels, out_channels): #Aquí se definen los canales de entrada y salida para cada bloque de convolución
        super().__init__() #Esto inicializa la clase padre nn.Module, lo cual es necesario para que nuestro bloque de convolución funcione correctamente dentro de la arquitectura de PyTorch.
        self.double_conv = nn.Sequential( #Aquí se define la secuencia de operaciones que componen el bloque de convolución doble. Se utiliza nn.Sequential para encadenar las operaciones de manera ordenada.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), #Primera capa de convolución 2D que toma in_channels y produce out_channels. El kernel_size=3 indica que se utiliza un filtro de 3x3, padding=1 asegura que la salida tenga el mismo tamaño espacial que la entrada, y bias=False significa que no se utilizará un término de sesgo en esta capa.
            nn.BatchNorm2d(out_channels), #Capa de normalización por lotes que normaliza las activaciones de la capa anterior. Esto ayuda a estabilizar el entrenamiento y acelerar la convergencia.
            nn.ReLU(inplace=True), #Función de activación ReLU que introduce no linealidad en el modelo. inplace=True permite que la operación se realice en el lugar, lo que puede ahorrar memoria.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): #El método forward define cómo se procesan los datos a través del bloque de convolución doble. Toma una entrada x y la pasa a través de la secuencia de operaciones definida en self.double_conv, devolviendo el resultado final.
        return self.double_conv(x)
 

class UNet48(nn.Module): 
    """
    U-Net de Dos bloques encoder y dos bloques decoder utilizada para Astronomía (48x48 píxeles).
    Contiene estrictamente DOS Encoders y DOS Decoders para un refinamiento progresivo.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        # =====================================================================
        # Estructura de la U-net: Entrada → [Encoder 1] → [Encoder 2] → [Bottleneck] → [Decoder 1] → [Decoder 2] → Salida
        # =====================================================================
        # ENCODER 1 (Recibe la imagen de entrada y se encarga de extraer características gruesas)
        self.enc1 = DoubleConv(in_channels, base_channels) #Se extraen caracteristicas gruesas, tensor de 64x48x48
        self.enc1_down = nn.MaxPool2d(2) #Bajada de resolución, tensor de 64x24x24
        #Encoder 2 (Recibe la salida de Encoder 1 y se encarga de extraer características más profundas para el refinamiento)
        self.enc2 = DoubleConv(base_channels, base_channels*2) #Se extraen caracteristicas más profundas, tensor de 128x24x24
        self.enc2_down = nn.MaxPool2d(2) #Bajada de resolución, tensor de 128x12x12
        #Bottleneck (El corazón de la U-Net, donde se extraen las características más abstractas y profundas de la imagen)
        self.bottleneck = DoubleConv(base_channels*2, base_channels*4) #Se extraen características aún más profundas, tensor de 256x12x12
        # DECODER 1 (Recibe la salida del Bottleneck y se encarga de reconstruir la imagen a una resolución intermedia)
        self.dec1_up = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2) #Subida de resolución, tensor de 128x24x24
        self.dec1 = DoubleConv(base_channels*4, base_channels*2) #Convolución para refinar la imagen, tensor de 128x24x24 (concatenación de la salida del Bottleneck y la salida de Encoder 2)
        # DECODER 2 
        self.dec2_up = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2) #Subida de resolución, tensor de 64x48x48
        self.dec2 = DoubleConv(base_channels*2, base_channels) #Convolución para refinar la imagen, tensor de 64x48x48
        self.dec2_final = nn.Sequential(nn.Conv2d(base_channels, out_channels, kernel_size=1),#Capa de salida que reduce los canales a out_channels (1 en este caso), tensor de 1x48x48
                                      nn.Tanh()) #tanh para asegurar que la salida esté en el rango [-1, 1] , consitente con la normalización de las imágenes de entrada
    def forward(self, x):
        # --- FLUJO ETAPA 1 --- 
        enc1_out = self.enc1(x) #Salida del Encoder 1, tensor de 64x48x48
        enc2_out = self.enc2(self.enc1_down(enc1_out)) #Salida del Encoder 2, tensor de 128x24x24
        b= self.bottleneck(self.enc2_down(enc2_out)) #Salida del Bottleneck, tensor de 256x12x12
        # --- FLUJO ETAPA 2 ---
        dec1_up= self.dec1_up(b) #Subida de resolución en Decoder 1, tensor de 128x24x24
        dec1_out = self.dec1(torch.cat([dec1_up, enc2_out], dim=1)) #Concatenación de la salida de Decoder 1 y la salida de Encoder
        dec2_up = self.dec2_up(dec1_out) #Subida de resolución en Decoder 2, tensor de 64x48x48
        dec2_out = self.dec2(torch.cat([dec2_up, enc1_out], dim=1)) #Concatenación de la salida de Decoder 2 y la salida de Encoder 1, tensor de 64x48x48
        
        # Resultado final deconvolucionado de alta calidad
        final_output = self.dec2_final(dec2_out) #tensor de 1x48x48, imagen de salida limpia con rango de valores [-1, 1]
        
        return final_output