# Changelog

All notable changes to this project will be documented in this file.

---

## [2.0.5] — 2026-02-07

### Added

- **MQTT Song Endpoint**: Added MQTT command support to play songs via the `billy/song` topic.
- **Factory Reset Options**: Added option in the settings to reset configuration. (.env file, user profiles .ini's, custom personas .ini's, logs, git, wifi, reboot)

### Changed

- **Persona Form**: Renamed the identity section and combined voice controls into the “Persona Voice & Expression” area.

### Fixed

- **Provider Startup**: Missing API keys no longer hard-fail startup; a warning is logged and services continue.
- **Voice Options Load**: Config endpoint now handles missing realtime providers without crashing.

---

## [2.0.4] — 2025-02-02

### Added

- **Realtime AI Providers**: Introduced a `RealtimeAIProvider` abstraction with a provider registry to support multiple realtime AI backends (contribution by: @turekaj)
- **XAI Provider (Beta)**: XAI support is currently beta and not fully integrated into the UI yet. For now, enable it by adding `REALTIME_AI_PROVIDER=xai` and `XAI_API_KEY=XAI API KEY HERE` to your `.env` file manually. (contribution by: @turekaj)
- **Startup Flap Toggle**: Added `FLAP_ON_BOOT` setting and UI control to enable the startup flap animation only when desired. (contribution by: @turekaj)

### Fixed

- **Installation Documentation**: Added missing `swig` dependency to installation instructions. The `lgpio` Python package requires `swig` to build from source, which was causing installation failures.

---

## [2.0.3] — 2025-01-09

### Added

- **Mockfish Development Mode**: Added `MOCKFISH` environment variable to enable development and testing without physical hardware. When enabled, GPIO operations and button presses are mocked, allowing full testing of Billy's functionality on any machine without a Raspberry Pi or physical Billy Bass. (contribution by: @turekaj)

### Fixed

- **Version Detection**: Fixed issue where the updater would show an incorrect version (e.g., v2.0.1) even when a newer tag (e.g., v2.0.2) was checked out. Improved version detection logic now properly handles detached HEAD state and uses multiple fallback methods (`git tag --points-at HEAD`, `git describe --tags --exact-match`, `git describe --tags`) to accurately detect the current checked-out version.
- **Logging Integration**: Version detection and update processes now use the centralized logging system, respecting the `LOG_LEVEL` setting for appropriate verbosity.

---

## [2.0.2] — 2025-12-25

### Fixed

- **Song Interruption**: Fixed critical bug where interrupting a song (especially after multiple times) would cause the session to hang indefinitely. Button presses after interruption would have no effect. The fix adds proper interrupt checking in the song playback loop and uses a timeout-based approach instead of blocking indefinitely.
- **Documentation**: Added missing `liblgpio-dev` and `liblgpio1` system dependencies to installation instructions. These packages are required for the `lgpio` Python package to build successfully.

---

## [2.0.1] — 2025-12-07

### Fixed

- **MQTT Session Hanging**: Fixed issue where MQTT announcements without follow-up (e.g., "The washing machine is finished") would leave the session open indefinitely until hitting OpenAI's 60-minute timeout limit. Sessions now properly close when no follow-up is needed.

---

## [2.0.0] — 2025-11-02
>### 🎉 Major New Features:
> User Profiles with memories - Multiple Personas - Custom Song Manager - Mqtt Say command support for follow-ups 

>### ⚠️ For existing builds of Billy
>
> **If you're coming from <= v1.4.0:** Please select the Legacy Pin Layout in the Hardware Settings section if you can't switch to the new unified wiring layout (see [BUILDME.md](./docs/BUILDME.md#from-motor-driver-to-raspberry-pi-gpio-pinout))

### Added

- **User Profiles**: Billy now identifies users when they introduce themselves (e.g., "Hi Billy, it's Thom")
  - Create/switch profiles, set display names, guest mode
  - Import/export profiles
  - Profile-specific statistics like interaction count and last seen timestamps
- **Memory System**: Billy remembers facts, preferences, and relationships for each user
  - Categorize by type (fact, preference, relationship, event, other)
  - Set importance levels (low, medium, high)
  - Edit/delete memories through UI
- **Multiple Personas**: Create and switch between different Billy personalities
  - Voices and mouth articulation per persona
  - Configurable personality traits (humor, sarcasm, honesty, etc.)
  - Import/export personas
  - Mid-conversation persona switching with graceful voice changes
  - Persona presets/templates for quick setup
- **Custom Song Manager**: Tooling for managing custom songs
  - Upload custom audio files (full.wav, vocals.wav, drums.wav)
  - Configure playback & animation settings (gain, tail threshold, compensate tail, head moves, half tempo tail flap)
  - Set song title and keywords for AI-triggered playback
  - Preview audio files before saving
  - Copy example songs to get started
- **UI Improvements**: 
  - 4-level configurable logging (ERROR/WARNING/INFO/VERBOSE)
  - Loading states and optimized polling

### Changed

- **Persona Storage**: Personas now stored in `./personas/persona_name/persona.ini`. The default persona remains at `./persona.ini` for backward compatibility.
- **Custom Wake-up Sounds**: Wake-up sounds are now saved per persona instead of globally
- **Logging System**: Replaced `DEBUG_MODE` with configurable `LOG_LEVEL` system (ERROR/WARNING/INFO/VERBOSE)
- **Mouth Articulation**: Now configured per persona instead of globally
- **Song Metadata Format**: Updated from `metadata.txt` to `metadata.ini` for consistency
- **Song Directory**: Custom songs now stored in `custom_songs/` (git-ignored) instead of `sounds/songs/` (which still holds the fishsticks song as a template)
- **MQTT Follow-up Handling**: MQTT prompts now respect `follow_up_intent` - if Billy asks a question via MQTT (e.g., "What's your favorite color?"), the mic stays open for your response in `auto` mode

### Fixed

- **Self-Triggering**: Billy no longer hears his own wake-up sounds
- **Head Movement**: Prevented head from getting stuck during routines
- **Session Hanging**: Added timeouts to prevent stuck sessions
- **Audio**: Standardized RMS calculations for consistent silence detection
- **Function Call Flow**: Fixed "conversation already has an active response" error by properly closing function calls with `function_call_output` before OpenAI auto-generates responses
- **GPIO Cleanup**: Fixed "double free or corruption" crash on shutdown by properly closing GPIO resources
- **Smart Home Commands**: Billy now correctly distinguishes between direct commands ("turn on lights") and questions ("ask if lights should be on") - no longer executes commands when asked to ask first
- **Function Confirmations**: Fixed issue where Billy wouldn't always provide spoken confirmation after smart home commands and personality changes - now reliably triggers verbal responses

---

## [1.5.0] — 2025-10-13

>### ⚠️ For existing builds of Billy: ⚠️ 
> 
> **Please select the Legacy Pin Layout in the Hardware Settings section if you can't switch to the new unified wiring layout (see [BUILDME.md](./docs/BUILDME.md#from-motor-driver-to-raspberry-pi-gpio-pinout))**

### Added
- **Configurable Pin Layouts:** Introduced `BILLY_PINS` Pin Layout setting (`new` / `legacy`) to switch between the new (default) pin layout and the legacy pin layout (for builds before october '25)
- **Mouth Articulation Control:** Added `MOUTH_ARTICULATION` (1–10) setting to fine-tune speech motion responsiveness.
- **Error Sound Handling:**  Centralized error playback — now plays `error.wav`, `noapikey.wav`, or `nowifi.wav` depending on the issue.
- **Release notes notification:** Notification with the release notes of the latest version.

### Changed
- **Unified GPIO Logic:** Refactored motor control for both Modern (2-motor) and Classic (3-motor) models into a single system. Default pin assignments moved to safer GPIOs. Unused H-bridge inputs are now grounded;
- **Movement Refinements:** Improved PWM handling and non-blocking motion timing for smoother, more natural flapping.
- **UI Enhancements:** Added Billy artwork to header and included reboot/power/restart-ui buttons. Improved feedback for mic and speaker device tests.

### Fixed
- Minor mouth sync inconsistencies under load.
- Occasional stalls caused by blocking PWM threads.
- Better recovery after OpenAI API or network errors.
- Motor watchdog will now disengage any motor that is on > 30 seconds

---

## [1.4.0] — 2025-09-04

### Added

- **Say command**: Added a MQTT command to let billy announce messages, either as literal sentences or as prompts.
- **Custom Wake-up Sounds**: Custom Wake-up sounds can now be customised and generated locally
- **New gpt-realtime model**: Added Support for the new stable release of the openAI Realtime API model.
- **Favicon**: No more 404

### Changed

- Improved update process by re-installing python requirements on software update
- Updated personality traits prompt to be more descriptive and more distinct.
- Disabled Flask debug mode by default.

---

## [1.3.1] — 2025-08-19

### Added

- **Shutdown and restart**: Added raspberry pi shutdown and restart controls (contribution by @cprasmu )

---

## [1.3.0] — 2025-07-28

### Added

- **'Dory' Mode**: Optional single-response mode where Billy answers only once before ending the session. (requested by @kenway33 )
- **Motor Test controls**: New motor test buttons in the Hardware tab allow triggering mouth and head/tail motion directly. (requested by @henrym9)
- **Hostname + Port Configuration**: Added settings for customizing the device's hostname and Flask port. (requested by @cprasmu)
- **Import/Export**: Added ability to upload/download both `.env` and `persona.ini` files.

### Changed

- Improved internal JS structure and modularization.
- Minor refinements to input label styling.
- Improved UX:
  - Added collapsible UI sections
  - **Reduced Motion Mode**: New UI toggle to disable animations and backdrop blur for accessibility or preference. Setting persists in local storage.
  - **Tooltips**: Informational tooltips added to multiple UI elements for better user guidance.

### Fixed

- **Motor Retract Fix**: Ensures Billy's head reliably returns to neutral after session ends.

---

## [1.2.0] — 2025-07-24

### Added

- Web-based user interface for easy configuration of Billy
  - Versioning check logic during boot & button to trigger OTA update
  - Speaker volume test and control in UI.
  - Tailwind CSS included locally for styling.
  - Password field visibility toggles in the UI.
  - Dropdown for selecting voice options in UI.
  - Mic input and speaker output level test utility.
- Option to change openAI Model
- Compatibilty for Classic Billy Model with 3 motors

### Changed

- Folder structure simplified and clarified.
- Automatic creation of `.env` and `persona.ini` from *.example files on first run.
- Committed `persona.ini`; now ignored by `.gitignore`.

### Added in beta

- MQTT "say" command integration for announcing messages
- Systemd service install process.
- Wi-Fi onboarding form (captive portal)
