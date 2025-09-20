from escpos.printer import Serial

""" 9600 Baud, 8N1, Flow Control Enabled """
p = Serial(devfile='COM9',
           baudrate=230400,
           bytesize=8,
           parity='N',
           stopbits=1,
           timeout=1.00,
           dsrdtr=True)

print("starting")
p._raw(bytes([29, 40, 75, 2, 0, 49, 69, 255]))
p.text("Hello World\n")
# p.qr("You can readme from your smartphone")
# p.image("example-lammas.png")
# p.image("ziv-snipped.png")
# p.image("zivd-snipped.png")
p.cut()

print("ended")