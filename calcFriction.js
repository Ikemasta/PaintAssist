export function calcFriction(L, D, e, QvFAD, P, ksi, Ta) {

    // P in bar
    // QvFAD in m3/min
    // e in mm

    const Rd = 287.058;

    const rho = P * 100000 / Rd / (Ta + 273.3);

    const QvFADs = QvFAD / 60.0;      // m3/s

    const Qv = QvFADs / P;            // actual volumetric flow

    const A = Math.PI * D * D / 4.0;

    const v = Qv / A;

    const nu = 2e-5 / rho;

    const Re = v * D / nu;

    const f =
        0.25 /
        Math.pow(
            Math.log10(
                (e / (D * 1000)) / 3.7 +
                5.74 / Math.pow(Re, 0.9)
            ),
            2
        );

    const DP =
        (ksi + f * L / D) *
        0.5 *
        P *
        1.2 *
        v * v /
        1e5;

    return {
        DPf: DP,
        v: v
    };
}
