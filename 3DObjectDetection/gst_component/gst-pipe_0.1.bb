inherit cmake

DESCRIPTION = "object tracking"
LICENSE = "BSD"
SECTION = "gstpipe"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/${LICENSE};md5=3775480a712fc46a69647678acb234cb"

# Dependencies.
DEPENDS := "opencv"
DEPENDS += "libnl zlib"
DEPENDS += "glib-2.0"
DEPENDS += "libutils"
DEPENDS += "libcutils"

DEPENDS += "gstreamer1.0"
DEPENDS += "gstreamer1.0-plugins-base"
DEPENDS += "gstreamer1.0-plugins-qti-oss-mlmeta"
DEPENDS += "gstreamer1.0-plugins-qti-oss-tools"
DEPENDS += "gstreamer1.0-rtsp-server"

FILESPATH =+ "${WORKSPACE}/example/gst-pipe/bin/:"

PN = "gst-pipe"
SRC_URI = "file://wifi_led/"
INSANE_SKIP_${PN}-dev += "ldflags dev-elf dev-deps"
PACKAGES = "${PN}-dbg ${PN} ${PN}-dev"
S = "${WORKSPACE}/example/gst-pipe/"

# Install directries.
INSTALL_INCDIR := "${includedir}"
INSTALL_BINDIR := "${bindir}"
INSTALL_LIBDIR := "${libdir}"

EXTRA_OECMAKE += ""

FILES_${PN} += "${INSTALL_BINDIR}"
FILES_${PN} += "${INSTALL_LIBDIR}"

SOLIBS = ".so*"
FILES_SOLIBSDEV = ""
