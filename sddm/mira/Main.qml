// =============================================================
// Mira — SDDM greeter theme (Qt 6).
// Copyright (C) 2026 Misaalya
// SPDX-License-Identifier: GPL-3.0-only
// =============================================================
// Matches the Mira website visual language: live 4x4 ordered-Bayer
// pixel-dithered underwater ruins GPU shader, tall architectural
// pixel-sans clock + twinkling pixel star, Inter UI, and clean
// minimalist footer.
//
// Keys:
//   Enter          log in           F1 / Shift+F1  next / prev session
//   Esc            clear password   F2 / Shift+F2  next / prev user
//   F3 / Shift+F3  cycle pixel palette (Mira / Matcha / Ember / Fjord / Aria)
// Power actions are mouse-only so a stray keypress never powers off.
import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    width: 1920
    height: 1080
    color: pal.bg

    // 5 curated Mira underwater pixel palettes
    readonly property var palettes: [
        {
            key: "mira",
            label: "Mira",
            bg: "#08090d",
            accent: "#d9d84a",
            accentInk: "#08090d",
            rayGain: 1.05,
            glint: "#e8eca8",
            bubble: "#d6f4f2",
            ramp: ["#050d1c", "#0a1b32", "#12314c", "#1e526c", "#347c90", "#5eaab2", "#9cd4d2", "#e2f0e8"]
        },
        {
            key: "matcha",
            label: "Matcha",
            bg: "#0b120e",
            accent: "#c5d988",
            accentInk: "#0d1410",
            rayGain: 1.00,
            glint: "#e0eeb6",
            bubble: "#dcf2d8",
            ramp: ["#06100d", "#0c1f19", "#15362b", "#245644", "#3e7e62", "#68a884", "#a4d0aa", "#e8f0d6"]
        },
        {
            key: "ember",
            label: "Ember",
            bg: "#100808",
            accent: "#f37c52",
            accentInk: "#140907",
            rayGain: 1.18,
            glint: "#ffd694",
            bubble: "#ffe0c0",
            ramp: ["#0d070e", "#1d0d16", "#38161e", "#622626", "#944231", "#c66c48", "#eea472", "#ffe0b6"]
        },
        {
            key: "fjord",
            label: "Fjord",
            bg: "#080f18",
            accent: "#95c2e2",
            accentInk: "#091018",
            rayGain: 1.02,
            glint: "#c6e8fc",
            bubble: "#d8eefc",
            ramp: ["#050b16", "#0b182c", "#152e4c", "#254c74", "#40729c", "#6a9ec4", "#a2cce6", "#e0f0fa"]
        },
        {
            key: "aria",
            label: "Aria",
            bg: "#07080f",
            accent: "#a6b4ff",
            accentInk: "#07080f",
            rayGain: 1.06,
            glint: "#bac4ff",
            bubble: "#c6d0f8",
            ramp: ["#05060c", "#0a0c18", "#12162a", "#202644", "#363e66", "#586292", "#909aca", "#dce2f8"]
        }
    ]

    function paletteIndexByKey(k) {
        if (!k) return 0
        const lk = String(k).toLowerCase()
        for (let i = 0; i < palettes.length; i++)
            if (palettes[i].key === lk) return i
        return 0
    }

    property int palIdx: paletteIndexByKey(config.palette)
    readonly property var pal: palettes[palIdx]
    readonly property color fg: "#ffffff"
    readonly property color fgSoft: Qt.rgba(0.92, 0.96, 1.0, 0.78)
    readonly property color fgDim: Qt.rgba(0.86, 0.92, 0.98, 0.56)
    readonly property color accent: pal.accent
    readonly property color accentInk: pal.accentInk
    readonly property string sans: config.font || "Inter"
    readonly property real s: Math.max(0.75, height / 1080)
    readonly property real cellSize: config.cellSize ? Math.max(2, Number(config.cellSize)) : 3.0
    readonly property bool useDeskWall: String(config.useDesktopWallpaper).toLowerCase() === "true"

    property bool busy: false
    property string message: ""
    property date now: new Date()
    property bool colonLit: true
    property real mouseNorm: 0.0
    property real failPulse: 0.0

    Behavior on mouseNorm { SmoothedAnimation { velocity: 2.2 } }

    HoverHandler {
        id: bgHover
        onPointChanged: {
            if (root.width > 0)
                root.mouseNorm = (point.position.x / root.width) * 2.0 - 1.0
        }
    }

    // ------------------------------------------------------ Background
    // Source texture sampled by the GPU ordered-Bayer pixel shader.
    Image {
        id: sceneImg
        visible: false
        asynchronous: false
        cache: false
        smooth: true
        source: root.useDeskWall && config.wallpaper ? ("file://" + config.wallpaper) : "assets/scene.jpg"
        onStatusChanged: {
            if (status === Image.Error && source !== "assets/scene.jpg")
                source = "assets/scene.jpg"
        }
    }

    // Pre-dithered fallback if GPU ShaderEffect is unavailable on legacy hardware.
    Image {
        anchors.fill: parent
        source: "assets/fallback-dither.png"
        fillMode: Image.PreserveAspectCrop
        smooth: false
        visible: oceanShader.status !== ShaderEffect.Compiled
    }

    ShaderEffect {
        id: oceanShader
        anchors.fill: parent
        visible: status === ShaderEffect.Compiled

        property variant source: sceneImg
        property vector2d uRes: Qt.vector2d(width, height)
        readonly property real vpA: width / Math.max(1, height)
        readonly property real imA: sceneImg.implicitWidth > 0
            ? (sceneImg.implicitWidth / Math.max(1, sceneImg.implicitHeight))
            : (16.0 / 9.0)
        property vector4d uCover: vpA > imA
            ? Qt.vector4d(1.0, imA / vpA, 0.0, (1.0 - imA / vpA) * 0.5)
            : Qt.vector4d(vpA / imA, 1.0, (1.0 - vpA / imA) * 0.5, 0.0)

        property real uTime: 1.6
        property real uCell: Math.max(2.0, Math.round(root.cellSize * root.s))
        property real uMouse: root.mouseNorm
        property real uRayGain: root.pal.rayGain
        property real uDive: root.busy ? 0.35 : 0.0
        property real uFail: root.failPulse
        property real uHasTex: sceneImg.status === Image.Ready ? 1.0 : 0.0

        property color uR0: root.pal.ramp[0]
        property color uR1: root.pal.ramp[1]
        property color uR2: root.pal.ramp[2]
        property color uR3: root.pal.ramp[3]
        property color uR4: root.pal.ramp[4]
        property color uR5: root.pal.ramp[5]
        property color uR6: root.pal.ramp[6]
        property color uR7: root.pal.ramp[7]
        property color uGlint: root.pal.glint
        property color uBubbleTint: root.pal.bubble

        Behavior on uDive { NumberAnimation { duration: 650; easing.type: Easing.OutCubic } }
        Behavior on uRayGain { NumberAnimation { duration: 480 } }
        Behavior on uR0 { ColorAnimation { duration: 480 } }
        Behavior on uR1 { ColorAnimation { duration: 480 } }
        Behavior on uR2 { ColorAnimation { duration: 480 } }
        Behavior on uR3 { ColorAnimation { duration: 480 } }
        Behavior on uR4 { ColorAnimation { duration: 480 } }
        Behavior on uR5 { ColorAnimation { duration: 480 } }
        Behavior on uR6 { ColorAnimation { duration: 480 } }
        Behavior on uR7 { ColorAnimation { duration: 480 } }
        Behavior on uGlint { ColorAnimation { duration: 480 } }
        Behavior on uBubbleTint { ColorAnimation { duration: 480 } }

        fragmentShader: "assets/underwater.frag.qsb"
    }

    FrameAnimation {
        running: true
        onTriggered: oceanShader.uTime += Math.min(0.05, frameTime)
    }

    // Website-matched subtle top/bottom & center reading scrim
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.00; color: Qt.rgba(0.015, 0.035, 0.075, 0.34) }
            GradientStop { position: 0.22; color: Qt.rgba(0.015, 0.035, 0.075, 0.04) }
            GradientStop { position: 0.50; color: Qt.rgba(0.015, 0.035, 0.075, config.dim ? Math.min(0.55, Number(config.dim)) : 0.28) }
            GradientStop { position: 0.80; color: Qt.rgba(0.015, 0.035, 0.075, 0.30) }
            GradientStop { position: 1.00; color: Qt.rgba(0.015, 0.035, 0.075, 0.64) }
        }
    }

    Timer {
        interval: 1000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: {
            root.now = new Date()
            root.colonLit = !root.colonLit
        }
    }

    // Hidden 1px lists exposing SDDM userModel and sessionModel roles.
    ListView {
        id: users
        width: 1; height: 1; opacity: 0
        model: userModel
        currentIndex: userModel.lastIndex >= 0 ? userModel.lastIndex : 0
        delegate: Item {
            property string login: model.name
            property string display: model.realName ? model.realName : model.name
        }
    }
    ListView {
        id: sessions
        width: 1; height: 1; opacity: 0
        model: sessionModel
        currentIndex: sessionModel.lastIndex >= 0 ? sessionModel.lastIndex : 0
        delegate: Item {
            property string label: model.name
        }
    }

    function cycle(view, step) {
        if (view.count > 0)
            view.currentIndex = (view.currentIndex + step + view.count) % view.count
    }
    function cyclePalette(step) {
        root.palIdx = (root.palIdx + step + root.palettes.length) % root.palettes.length
    }
    function login() {
        if (busy || !users.currentItem)
            return
        busy = true
        message = ""
        sddm.login(users.currentItem.login, password.text, sessions.currentIndex)
    }

    Connections {
        target: sddm
        function onLoginFailed() {
            root.busy = false
            root.message = "Incorrect password"
            password.text = ""
            password.forceActiveFocus()
            shake.restart()
            failAnim.restart()
        }
    }

    SequentialAnimation {
        id: failAnim
        NumberAnimation { target: root; property: "failPulse"; to: 1.0; duration: 90 }
        NumberAnimation { target: root; property: "failPulse"; to: 0.0; duration: 520; easing.type: Easing.OutCubic }
    }

    // --------------------------------------------------- Top Minimal Bar
    Item {
        visible: primaryScreen
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: Math.round(28 * root.s)
        anchors.leftMargin: Math.round(36 * root.s)
        anchors.rightMargin: Math.round(36 * root.s)
        height: Math.round(34 * root.s)

        // Top-right: quiet hostname + Caps Lock warning only when active
        Row {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            spacing: Math.round(14 * root.s)

            Rectangle {
                visible: keyboard.capsLock
                anchors.verticalCenter: parent.verticalCenter
                height: Math.round(24 * root.s)
                width: capsTxt.implicitWidth + Math.round(18 * root.s)
                radius: Math.round(6 * root.s)
                color: Qt.rgba(0.03, 0.05, 0.09, 0.72)
                border.width: 1
                border.color: root.accent

                Text {
                    id: capsTxt
                    anchors.centerIn: parent
                    color: root.accent
                    font.family: root.sans
                    font.pixelSize: Math.round(11 * root.s)
                    font.weight: Font.DemiBold
                    font.letterSpacing: 0.6 * root.s
                    text: "CAPS LOCK"
                }
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: root.fgDim
                font.family: root.sans
                font.pixelSize: Math.round(13 * root.s)
                font.weight: Font.Medium
                text: sddm.hostName
            }
        }
    }

    // ------------------------------------------------------ Optical Center
    Column {
        visible: primaryScreen
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -Math.round(24 * root.s)
        spacing: 0

        // Architectural pixel-sans clock + perched twinkling pixel star
        Item {
            anchors.horizontalCenter: parent.horizontalCenter
            width: clock.width
            height: clock.height

            PixelClock {
                id: clock
                anchors.centerIn: parent
                px: Math.max(5, Math.round(8 * root.s))
                timeText: Qt.formatDateTime(root.now, "HH:mm")
                ink: root.fg
                colonInk: root.accent
                colonLit: root.busy ? true : root.colonLit
            }

            PixelStar {
                anchors.left: clock.right
                anchors.leftMargin: -Math.round(4 * root.s)
                anchors.top: clock.top
                anchors.topMargin: -Math.round(22 * root.s)
                unit: Math.max(2, Math.round(4 * root.s))
                tint: root.accent
                fast: root.busy
            }
        }

        Item { width: 1; height: Math.round(18 * root.s) }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            color: root.fgSoft
            font.family: root.sans
            font.pixelSize: Math.round(16 * root.s)
            font.weight: Font.Medium
            font.letterSpacing: -0.2 * root.s
            text: Qt.formatDateTime(root.now, "dddd, d MMMM")
        }

        Item { width: 1; height: Math.round(36 * root.s) }

        // Minimal dark-glass password field matching Mira's UI & website
        Rectangle {
            id: box
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.round(344 * root.s)
            height: Math.round(46 * root.s)
            radius: Math.round(10 * root.s)
            color: Qt.rgba(0.025, 0.045, 0.085, 0.76)
            border.width: 1
            border.color: root.failPulse > 0.1
                ? "#f06565"
                : (password.text.length > 0 ? root.accent : (password.activeFocus ? Qt.rgba(1, 1, 1, 0.30) : Qt.rgba(1, 1, 1, 0.16)))
            transform: Translate { id: nudge }
            Behavior on border.color { ColorAnimation { duration: 160 } }

            Image {
                id: lock
                anchors.left: parent.left
                anchors.leftMargin: Math.round(15 * root.s)
                anchors.verticalCenter: parent.verticalCenter
                width: Math.round(17 * root.s)
                height: width
                sourceSize: Qt.size(width * 2, height * 2)
                source: "assets/lock.svg"
                opacity: password.activeFocus ? 0.92 : 0.62
                Behavior on opacity { NumberAnimation { duration: 140 } }
            }

            TextInput {
                id: password
                anchors.left: lock.right
                anchors.leftMargin: Math.round(11 * root.s)
                anchors.right: submitBtn.left
                anchors.rightMargin: Math.round(10 * root.s)
                anchors.verticalCenter: parent.verticalCenter
                clip: true
                echoMode: TextInput.Password
                passwordCharacter: "•"
                inputMethodHints: Qt.ImhSensitiveData | Qt.ImhNoPredictiveText | Qt.ImhNoAutoUppercase
                color: root.fg
                selectionColor: root.accent
                selectedTextColor: root.accentInk
                font.family: root.sans
                font.pixelSize: Math.round(15 * root.s)
                font.weight: Font.Medium
                font.letterSpacing: text.length ? 3 * root.s : 0
                enabled: !root.busy
                focus: true
                Keys.onReturnPressed: root.login()
                Keys.onEnterPressed: root.login()
                Keys.onEscapePressed: { text = ""; root.message = "" }
                onTextEdited: root.message = ""

                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    visible: password.text.length === 0 && !password.preeditText
                    color: Qt.rgba(0.86, 0.92, 0.98, 0.44)
                    font.family: root.sans
                    font.pixelSize: Math.round(14 * root.s)
                    font.weight: Font.Normal
                    text: users.currentItem ? ("Password for " + users.currentItem.display) : "Password"
                }
            }

            // Crisp inline submit button (illuminates in accent when typing)
            Rectangle {
                id: submitBtn
                anchors.right: parent.right
                anchors.rightMargin: Math.round(7 * root.s)
                anchors.verticalCenter: parent.verticalCenter
                width: Math.round(32 * root.s)
                height: Math.round(32 * root.s)
                radius: Math.round(7 * root.s)
                color: password.text.length > 0 && !root.busy
                    ? root.accent
                    : Qt.rgba(1, 1, 1, submitArea.containsMouse ? 0.14 : 0.08)
                Behavior on color { ColorAnimation { duration: 160 } }

                Image {
                    anchors.centerIn: parent
                    width: Math.round(14 * root.s)
                    height: width
                    sourceSize: Qt.size(width * 2, height * 2)
                    source: password.text.length > 0 && !root.busy ? "assets/arrow-right.svg" : "assets/arrow-right-light.svg"
                    opacity: password.text.length > 0 && !root.busy ? 1.0 : 0.65
                }

                MouseArea {
                    id: submitArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.login()
                }
            }

            MouseArea {
                anchors.left: parent.left
                anchors.right: submitBtn.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                cursorShape: Qt.IBeamCursor
                onClicked: password.forceActiveFocus()
            }
        }

        Item { width: 1; height: Math.round(14 * root.s) }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            height: Math.round(18 * root.s)
            color: root.message !== "" ? "#ff827a" : root.fgSoft
            font.family: root.sans
            font.pixelSize: Math.round(13 * root.s)
            font.weight: Font.Medium
            text: root.message !== "" ? root.message : (root.busy ? "Logging in…" : "")
        }
    }

    SequentialAnimation {
        id: shake
        NumberAnimation { target: nudge; property: "x"; to: -10 * root.s; duration: 48 }
        NumberAnimation { target: nudge; property: "x"; to: 8 * root.s; duration: 64 }
        NumberAnimation { target: nudge; property: "x"; to: -5 * root.s; duration: 54 }
        NumberAnimation { target: nudge; property: "x"; to: 0; duration: 54 }
    }

    // --------------------------------------------------- Reusable Controls
    component Keycap: Rectangle {
        property alias label: cap.text
        width: cap.implicitWidth + Math.round(11 * root.s)
        height: Math.round(20 * root.s)
        radius: Math.round(5 * root.s)
        color: Qt.rgba(1, 1, 1, 0.06)
        border.width: 1
        border.color: Qt.rgba(1, 1, 1, 0.24)
        Text {
            id: cap
            anchors.centerIn: parent
            color: root.fg
            font.family: root.sans
            font.pixelSize: Math.round(10 * root.s)
            font.weight: Font.Bold
        }
    }

    component FooterAction: Item {
        id: act
        property string key: ""
        property string iconSrc: ""
        property string label: ""
        property bool active: true
        signal clicked
        implicitWidth: actRow.implicitWidth
        implicitHeight: Math.max(Math.round(24 * root.s), actRow.implicitHeight)
        opacity: active ? 1.0 : 0.42

        Row {
            id: actRow
            anchors.verticalCenter: parent.verticalCenter
            spacing: Math.round(7 * root.s)

            Keycap {
                label: act.key
                visible: act.key !== ""
                anchors.verticalCenter: parent.verticalCenter
            }
            Image {
                visible: act.iconSrc !== ""
                anchors.verticalCenter: parent.verticalCenter
                width: Math.round(14 * root.s)
                height: width
                sourceSize: Qt.size(width * 2, height * 2)
                source: act.iconSrc
                opacity: actArea.containsMouse ? 1.0 : 0.62
                Behavior on opacity { NumberAnimation { duration: 130 } }
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: actArea.containsMouse ? root.fg : root.fgSoft
                font.family: root.sans
                font.pixelSize: Math.round(12 * root.s)
                font.weight: Font.Medium
                text: act.label
                Behavior on color { ColorAnimation { duration: 130 } }
            }
        }
        MouseArea {
            id: actArea
            anchors.fill: parent
            hoverEnabled: true
            enabled: act.active
            cursorShape: Qt.PointingHandCursor
            onClicked: act.clicked()
        }
    }

    // --------------------------------------------------- Minimal Bottom Bar
    Item {
        visible: primaryScreen
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.leftMargin: Math.round(36 * root.s)
        anchors.rightMargin: Math.round(36 * root.s)
        anchors.bottomMargin: Math.round(26 * root.s)
        height: Math.round(28 * root.s)

        // Left: User & Session switchers
        Row {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            spacing: Math.round(18 * root.s)

            FooterAction {
                anchors.verticalCenter: parent.verticalCenter
                key: users.count > 1 ? "F2" : ""
                label: users.currentItem ? users.currentItem.display : ""
                active: users.count > 1
                opacity: 0.90
                onClicked: {
                    root.cycle(users, 1)
                    password.text = ""
                }
            }

            Rectangle {
                anchors.verticalCenter: parent.verticalCenter
                width: 1
                height: Math.round(12 * root.s)
                color: Qt.rgba(1, 1, 1, 0.22)
            }

            FooterAction {
                anchors.verticalCenter: parent.verticalCenter
                key: "F1"
                label: sessions.currentItem ? sessions.currentItem.label : "Session"
                active: sessions.count > 1
                onClicked: root.cycle(sessions, 1)
            }
        }

        // Right: Mouse-only power actions
        Row {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            spacing: Math.round(20 * root.s)

            FooterAction {
                anchors.verticalCenter: parent.verticalCenter
                iconSrc: "assets/suspend.svg"
                label: "Suspend"
                visible: sddm.canSuspend
                onClicked: sddm.suspend()
            }
            FooterAction {
                anchors.verticalCenter: parent.verticalCenter
                iconSrc: "assets/reboot.svg"
                label: "Reboot"
                visible: sddm.canReboot
                onClicked: sddm.reboot()
            }
            FooterAction {
                anchors.verticalCenter: parent.verticalCenter
                iconSrc: "assets/power.svg"
                label: "Power off"
                visible: sddm.canPowerOff
                onClicked: sddm.powerOff()
            }
        }
    }

    // --------------------------------------------------- Keyboard Shortcuts
    Keys.onPressed: (event) => {
        const back = event.modifiers & Qt.ShiftModifier ? -1 : 1
        if (event.key === Qt.Key_F1) {
            root.cycle(sessions, back)
        } else if (event.key === Qt.Key_F2) {
            root.cycle(users, back)
            password.text = ""
        } else if (event.key === Qt.Key_F3) {
            root.cyclePalette(back)
        } else {
            return
        }
        event.accepted = true
    }

    Component.onCompleted: password.forceActiveFocus()
}
