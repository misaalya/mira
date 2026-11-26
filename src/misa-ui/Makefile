# misa-ui: the rice's GTK3 popups in C (formerly Python). `make OUT=dir`
# builds every tool into dir; tools/deploy.sh does that into bin/.
CC ?= gcc
CFLAGS ?= -O2 -flto=auto -pipe -Wall -Wextra -Wno-unused-parameter -Wl,-s -Wl,--as-needed
PKG = gtk+-3.0
OUT ?= build
TOOLS = ClipPanel WindowSwitch MediaPillUI CardPicker WaitCompositor WifiPanel BluetoothPanel NotifPanel SoundPanel MusicPanel

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

$(OUT)/CardPicker: card-picker.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG) gdk-x11-3.0 x11) -o $@ card-picker.c common.c \
		$(shell pkg-config --libs $(PKG) gdk-x11-3.0 x11) -lm

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

$(OUT)/SoundPanel: sound-panel.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ sound-panel.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm

$(OUT)/MusicPanel: music-panel.c common.c common.h
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(shell pkg-config --cflags $(PKG)) -o $@ music-panel.c common.c \
		$(shell pkg-config --libs $(PKG)) -lm
