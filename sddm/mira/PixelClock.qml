// Architectural tall pixel-sans digital clock matching the Mira wordmark proportions.
import QtQuick

Canvas {
    id: clockCanvas
    property string timeText: "00:00"
    property color ink: "#ffffff"
    property color colonInk: "#d9d84a"
    property int px: 7
    property bool colonLit: true

    // Each digit is 7x12 units with 2px stroke thickness; colon is 2 units wide; gap is 2 units.
    readonly property int digitW: 7
    readonly property int digitH: 12
    readonly property int colonW: 2
    readonly property int gap: 2
    readonly property int totalUnitsW: (4 * digitW + colonW + 4 * gap)

    width: totalUnitsW * px
    height: digitH * px

    onTimeTextChanged: requestPaint()
    onInkChanged: requestPaint()
    onColonInkChanged: requestPaint()
    onPxChanged: requestPaint()
    onColonLitChanged: requestPaint()

    function drawDigit(ctx, ch, ox, u) {
        // Architectural 7x12 geometric pixel digits with chamfered outer corners (echoing MIRA)
        const r = (x, y, w, h) => ctx.fillRect((ox + x) * u, y * u, w * u, h * u)
        switch (ch) {
        case "0":
            r(1, 0, 5, 2); r(1, 10, 5, 2)
            r(0, 1, 2, 10); r(5, 1, 2, 10)
            break
        case "1":
            r(3, 0, 2, 12)
            r(1, 2, 2, 2); r(2, 1, 1, 2)
            r(1, 10, 6, 2)
            break
        case "2":
            r(1, 0, 5, 2); r(0, 1, 2, 2); r(5, 1, 2, 4)
            r(2, 5, 4, 2); r(0, 6, 3, 2); r(0, 7, 2, 5)
            r(0, 10, 7, 2)
            break
        case "3":
            r(1, 0, 5, 2); r(5, 1, 2, 4)
            r(2, 5, 4, 2)
            r(5, 7, 2, 4); r(1, 10, 5, 2)
            r(0, 1, 2, 2); r(0, 9, 2, 2)
            break
        case "4":
            r(0, 0, 2, 7); r(5, 0, 2, 12)
            r(0, 5, 7, 2)
            break
        case "5":
            r(0, 0, 7, 2); r(0, 0, 2, 6)
            r(0, 5, 6, 2); r(5, 6, 2, 5)
            r(1, 10, 5, 2); r(0, 9, 2, 2)
            break
        case "6":
            r(1, 0, 5, 2); r(0, 1, 2, 10)
            r(0, 5, 6, 2); r(5, 6, 2, 5)
            r(1, 10, 5, 2)
            break
        case "7":
            r(0, 0, 7, 2); r(5, 0, 2, 4)
            r(4, 4, 2, 3); r(3, 7, 2, 5)
            break
        case "8":
            r(1, 0, 5, 2); r(1, 5, 5, 2); r(1, 10, 5, 2)
            r(0, 1, 2, 4); r(5, 1, 2, 4)
            r(0, 7, 2, 4); r(5, 7, 2, 4)
            break
        case "9":
            r(1, 0, 5, 2); r(0, 1, 2, 5); r(5, 1, 2, 10)
            r(1, 5, 5, 2); r(1, 10, 5, 2)
            break
        }
    }

    onPaint: {
        const ctx = getContext("2d")
        ctx.clearRect(0, 0, width, height)
        const u = px
        const s = timeText.length === 5 ? timeText : "00:00"
        const offsets = [0, digitW + gap, 2 * (digitW + gap), 2 * (digitW + gap) + colonW + gap, 3 * (digitW + gap) + colonW + gap]

        // Soft ambient pixel drop shadow for readability over bright water caustics
        ctx.fillStyle = "rgba(4, 10, 20, 0.42)"
        ctx.save()
        ctx.translate(0, Math.max(2, Math.round(u * 0.45)))
        drawDigit(ctx, s[0], offsets[0], u)
        drawDigit(ctx, s[1], offsets[1], u)
        drawDigit(ctx, s[3], offsets[3], u)
        drawDigit(ctx, s[4], offsets[4], u)
        ctx.restore()

        ctx.fillStyle = String(ink)
        drawDigit(ctx, s[0], offsets[0], u)
        drawDigit(ctx, s[1], offsets[1], u)
        drawDigit(ctx, s[3], offsets[3], u)
        drawDigit(ctx, s[4], offsets[4], u)

        ctx.fillStyle = colonLit ? String(colonInk) : "rgba(255, 255, 255, 0.24)"
        ctx.fillRect(offsets[2] * u, 3 * u, 2 * u, 2 * u)
        ctx.fillRect(offsets[2] * u, 8 * u, 2 * u, 2 * u)
    }
}
