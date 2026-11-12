-- Builds ggml's CPU backend into the package's target directory and, because
-- lde skips Teal compilation when a build script is present, compiles src into
-- the same directory.

local build = require("lde-build")
local tl = require("tl")

local COMMIT = "57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234"
local CACHE = "../ggml-cache"
local SOURCE = CACHE .. "/src/ggml-" .. COMMIT

if not build:exists(SOURCE .. "/CMakeLists.txt") then
	build:write(
		"ggml.tar.gz",
		build:fetch("https://github.com/ggml-org/ggml/archive/" .. COMMIT .. ".tar.gz")
	)
	build:extract("ggml.tar.gz", CACHE .. "/src")
	build:delete("ggml.tar.gz")
end

build:sh(
	table.concat({
		"cmake",
		"-S " .. SOURCE,
		"-B " .. CACHE .. "/build",
		"-DCMAKE_BUILD_TYPE=Release",
		"-DBUILD_SHARED_LIBS=ON",
		"-DGGML_BUILD_TESTS=OFF",
		"-DGGML_BUILD_EXAMPLES=OFF",
		"-DGGML_METAL=OFF",
		"-DGGML_ACCELERATE=OFF",
		"-DGGML_BLAS=OFF",
		"-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
		"-DCMAKE_INSTALL_RPATH='$ORIGIN'",
		'-DCMAKE_INSTALL_PREFIX="' .. build.outDir .. "/" .. CACHE .. '/install"',
	}, " ")
)

build:sh("cmake --build " .. CACHE .. "/build --target install --parallel")
build:sh("rm -f libggml*.so*")
build:sh("cp -P " .. CACHE .. "/install/lib/libggml*.so* .")

for _, file in ipairs(build:scan(".", "**/*.tl")) do
	if not file:match("%.d%.tl$") then
		build:write((file:gsub("%.tl$", ".lua")), tl.gen(build:read(file)))
	end
end
