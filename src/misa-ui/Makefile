# misa-ui: the rice's GTK3 popups in C (formerly Python). `make OUT=dir`
# builds every tool into dir; tools/deploy.sh does that into bin/.
CC ?= gcc
CFLAGS ?= -O2 -pipe -Wall -Wextra -Wno-unused-parameter
PKG = gtk+-3.0
OUT ?= build
TOOLS = ClipPanel WindowSwitch MediaPillUI CardPicker WaitCompositor WifiPanel BluetoothPanel NotifPanel

all: $(addprefix $(OUT)/,$(TOOLS))

$(OUT)/ClipPanel: clip-panel.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ clip-panel.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm

$(OUT)/WindowSwitch: window-switch.c json.c json.h common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG) gdk-x11-3.0) -o $@ window-switch.c json.c common.c \
		$(shell pkg-config --libs $(PKG) gdk-x11-3.0) -lm

$(OUT)/MediaPillUI: media-pill.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ media-pill.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm

$(OUT)/CardPicker: card-picker.c live-gl.c live-gl.h common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ card-picker.c live-gl.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm -ldl

$(OUT)/WifiPanel: wifi-panel.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ wifi-panel.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm

$(OUT)/WaitCompositor: wait-compositor.c
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ wait-compositor.c $(shell pkg-config --libs x11)

clean:
	rm -rf build
.PHONY: all clean

$(OUT)/BluetoothPanel: bt-panel.c common.c common.h
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ bt-panel.c common.c $(shell pkg-config --libs $(PKG)) -lm

$(OUT)/NotifPanel: notif-panel.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG) gio-2.0) -o $@ notif-panel.c common.c \
		$(shell pkg-config --libs $(PKG) gio-2.0) -lm
