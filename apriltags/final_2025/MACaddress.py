from getmac import get_mac_address as gma


def GetJetson():
    MACdict = {
        "48:b0:2d:c1:63:9c" : "Jetson 1",
        "avocado" : "bagget"
    }
    return gma()
    print(MACdict[gma()])
    
print(GetJetson())
