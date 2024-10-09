from .QRCodeDetection.QrCodeScanner import QrCodeScanner

def initialize(camera: str):
    QrCodeScanner(camera)
