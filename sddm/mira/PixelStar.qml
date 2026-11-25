// Mira 4-point pixel star (exact 3-frame twinkle matching the Mira logo).
import QtQuick

Canvas {
    id: star
    property color tint: "#d9d84a"
    property int unit: 3
    property bool fast: false
    property int phase: 0

    width: 7 * unit
    height: 9 * unit

    onTintChanged: requestPaint()
    onUnitChanged: requestPaint()
    onPhaseChanged: requestPaint()

    onPaint: {
        const ctx = getContext("2d")
        ctx.clearRect(0, 0, width, height)
        ctx.fillStyle = String(tint)
        const u = unit
        if (phase === 0) {
            ctx.fillRect(3 * u, 0, u, 9 * u)
            ctx.fillRect(2 * u, 3 * u, 3 * u, 3 * u)
            ctx.fillRect(0, 4 * u, 7 * u, u)
        } else if (phase === 1) {
            ctx.fillRect(3 * u, 2 * u, u, 5 * u)
            ctx.fillRect(2 * u, 3 * u, 3 * u, 3 * u)
            ctx.fillRect(1 * u, 4 * u, 5 * u, u)
        } else {
            ctx.fillRect(3 * u, 3 * u, u, 3 * u)
            ctx.fillRect(2 * u, 4 * u, 3 * u, u)
        }
    }

    SequentialAnimation {
        running: true
        loops: Animation.Infinite
        PauseAnimation { duration: star.fast ? 110 : 4600 }
        PropertyAction { target: star; property: "phase"; value: 1 }
        PauseAnimation { duration: 120 }
        PropertyAction { target: star; property: "phase"; value: 2 }
        PauseAnimation { duration: 120 }
        PropertyAction { target: star; property: "phase"; value: 1 }
        PauseAnimation { duration: 120 }
        PropertyAction { target: star; property: "phase"; value: 0 }
    }
}
