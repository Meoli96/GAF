iter = 1
eps = 0.100000034E-05

# Condizioni Iniziali Stato 1
rho1 = 3.469732575
u1 = 416.667
p1 = 454700
gam1 = 1.4

# Condizioni Iniziali Stato 4
rho4 = 1.17622348246
u4 = 0
p4 = 100000
gam4 = 1.4

# Grandezze derivate
delta1 = 0.5 * (gam1 - 1)
delta4 = 0.5 * (gam4 - 1)
kappa1 = 0.5 * (gam1 + 1)
kappa4 = 0.5 * (gam4 + 1)
beta1 = gam1 / delta1
beta4 = gam4 / delta4
a1 = (gam1 * p1 / rho1) ** 0.5
a4 = (gam4 * p4 / rho4) ** 0.5

# Proprietà medie
gam = 0.5 * (gam1 + gam4)
deltam = 0.5 * (gam - 1)
alfam = deltam / gam

# Calcolo velocità di primo tentativo
z = (a4 / a1) * ((p1 / p4) ** alfam)
vel = (z * (a1 / deltam + u1) - (a4 / deltam - u4)) / (1 + z)
print(f'Velocità di primo tentativo = {vel:.3f} m/s')

# Inizio procedura Newton-Rhapson
while True:
    # ONDA U-A
    if vel <= u1:
        # Urto
        x1 = kappa1 * (u1 - vel) / (2 * a1)
        M1r = x1 + (1 + x1 ** 2) ** 0.5
        M1rq = M1r ** 2
        p2 = p1 * (1 + gam1 * (M1rq - 1) / kappa1)
        dp2 = -2 * gam1 * p1 * M1r / (a1 * (1 / M1rq + 1))
    else:
        # Espansione
        a2 = a1 - delta1 * (vel - u1)
        p2 = p1 * (a2 / a1) ** beta1
        dp2 = -gam1 * p2 / a2

    # ONDA U+A

    if vel >= u4:
        # URTO
        x4 = kappa4 * (u4 - vel) / (2 * a4)
        M4r = x4 - (1 + x4 ** 2) ** 0.5
        M4rq = M4r ** 2
        p3 = p4 * (1 + gam4 * (M4rq - 1) / kappa4)
        dp3 = -2 * gam4 * p4 * M4r / (a4 * (1 / M4rq + 1))
    else:
        # ESPANSIONE
        a3 = a4 + delta4 * (vel - u4)
        p3 = p4 * (a3 / a4) ** beta4
        dp3 = gam4 * p3 / a3

    # Controllo convergenza
    if abs(1 - p2 / p3) > eps:
        vel = vel - (p2 - p3) / (dp2 - dp3)
        iter = iter + 1
    else:
        break

# Calcolo variabili interfaccia

# velocità stati intermedi positiva interfaccia a sinistra d.c.
# Il caso v = 0 non è considerato e manda in errore
if vel > 0:
    gammaf = gam1
    uf = u1
    pf = p1
    rhof = rho1

    # velocità zona 2 minore velocità zona 1 = urto
    if vel < u1:
        # velocità urto negativa interfaccia zona 2
        w2 = u1 - a1 * M1r
        if w2 < 0:
            pf = p2
            uf = vel
            rhof = rho1 * (kappa1 / (delta1 + 1 / M1rq))
    else:
        # velocità zona 2 maggiore velocità zona 1 = espansione

        # lambda1 positivo interfaccia in zona 1
        lambda1 = u1 - a1
        if lambda1 < 0:
            pf = p2
            uf = vel
            rhof = -dp2 / a2

            # lambda2 maggiore zero flusso transonico

            lambda2 = vel - a2
            if lambda2 > 0:
                af = (a1 + delta1 * u1) / (1 + delta1)
                uf = af
                pf = p1 * (af / a1) ** beta1
                rhof = gam1 * pf / (af ** 2)

else:
    # velocità stati intermedi negativa interfaccia a destra d.c.
    gammaf = gam4
    uf = u4
    pf = p4
    rhof = rho4

    # velocità zona 3 maggiore velocità zona 4 = urto
    if vel > u4:
        w3 = u4 - a4 * M4r

        # velocità urto w3 positiva interfaccia zona 3

        if w3 > 0:
            uf = vel
            pf = p3
            rhof = rho4 * (kappa4 / (delta4 + 1 / M4rq))
    else:
        # velocità zona 3 minore velocità zona 4 = espansione

        # lambda4 negativa interfaccia zona 4
        lambda4 = u4 + a4
        if lambda4 > 0:
            uf = vel
            pf = p3
            rhof = dp3 / a3

            # lambda3 negativa espansione transonica

            lambda3 = vel + a3
            if lambda3 < 0:
                af = (a4 - delta4 * u4) / (1 + delta4)
                uf = -af
                pf = p4 * (af / a4) ** beta4
                rhof = gam4 * pf / (af ** 2)

# Calcolo flussi interfaccia
fmassa = rhof * uf
fqdm = pf + fmassa * uf
fe = uf * ((gammaf * pf / (gammaf - 1)) + fmassa * uf * 0.5)

print(f'Iterazioni = {iter}')
print(f'P2 = {p2/100000:.3f} bar')
print(f'P3 = {p3/100000:.3f} bar')
print(f'u3 = u4 = {vel:.3f} m/s')