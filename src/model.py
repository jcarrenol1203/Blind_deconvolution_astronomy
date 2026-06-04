import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Bloque básico de convolución doble (Conv + BatchNorm + ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class DualStageUNet48(nn.Module):
    """
    U-Net de Dos Etapas optimizada para Astronomía (48x48 píxeles).
    Contiene estrictamente DOS Encoders y DOS Decoders para un refinamiento progresivo.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # =====================================================================
        #  ETAPA 1: DECONVOLUCIÓN GRUESA (Primer Encoder y Primer Decoder)
        # =====================================================================
        # ENCODER 1
        self.enc1_inc = DoubleConv(in_channels, 32)
        self.enc1_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))   # 24x24
        self.enc1_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))  # 12x12
        self.enc1_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # 6x6
        
        # DECODER 1
        self.dec1_up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv1 = DoubleConv(256, 128)
        self.dec1_up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv2 = DoubleConv(128, 64)
        self.dec1_up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1_conv3 = DoubleConv(64, 32)
        self.dec1_out = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=1), nn.Tanh())

        # =====================================================================
        #  ETAPA 2: REFINAMIENTO FINO (Segundo Encoder y Segundo Decoder)
        # =====================================================================
        # ENCODER 2 (Recibe la salida de la Etapa 1)
        self.enc2_inc = DoubleConv(out_channels, 32)
        self.enc2_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.enc2_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.enc2_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # DECODER 2
        self.dec2_up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv1 = DoubleConv(256, 128)
        self.dec2_up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv2 = DoubleConv(128, 64)
        self.dec2_up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2_conv3 = DoubleConv(64, 32)
        self.dec2_out = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x):
        # --- FLUJO ETAPA 1 ---
        x1 = self.enc1_inc(x)
        x2 = self.enc1_down1(x1)
        x3 = self.enc1_down2(x2)
        x4 = self.enc1_down3(x3)
        
        u1 = torch.cat([self.dec1_up1(x4), x3], dim=1)
        u1 = self.dec1_conv1(u1)
        u2 = torch.cat([self.dec1_up2(u1), x2], dim=1)
        u2 = self.dec1_conv2(u2)
        u3 = torch.cat([self.dec1_up3(u2), x1], dim=1)
        u3 = self.dec1_conv3(u3)
        
        # Resultado intermedio (Predicción gruesa)
        coarse_output = self.dec1_out(u3)

        # --- FLUJO ETAPA 2 (Refinamiento) ---
        # Metemos la salida de la Etapa 1 al segundo juego de Encoder/Decoder
        y1 = self.enc2_inc(coarse_output)
        y2 = self.enc2_down1(y1)
        y3 = self.enc2_down2(y2)
        y4 = self.enc2_down3(y3)
        
        w1 = torch.cat([self.dec2_up1(y4), y3], dim=1)
        w1 = self.dec2_conv1(w1)
        w2 = torch.cat([self.dec2_up2(w1), y2], dim=1)
        w2 = self.dec2_conv2(w2)
        w3 = torch.cat([self.dec2_up3(w2), y1], dim=1)
        w3 = self.dec2_conv3(w3)
        
        # Resultado final deconvolucionado de alta calidad
        final_output = self.dec2_out(w3)
        
        return final_output