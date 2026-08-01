package com.interlekt.slmengine

import android.os.Debug
import android.util.Log
import java.io.File

/**
 * Per-question hardware instrumentation, read from /proc and /sys.
 *
 * Deliberately Context-free so EvalRunner's constructor stays unchanged.
 * Everything here is readable by an unprivileged app on a stock device; each
 * reader degrades to a sentinel rather than throwing, because sysfs layout
 * varies by SoC and vendor.
 *
 * The metric that matters most is cpuTimeMs — the app's own CPU time from
 * /proc/self/stat. Unlike wall-clock latency it is unaffected by cooldown
 * delays and background load. CPU-seconds per query is a defensible compute
 * proxy: on a CPU-only inference path, compute effort is dominated by core-seconds.
 */
object HardwareMonitor {

    private const val TAG = "SLMHw"

    /** Linux/Android USER_HZ. 100 on every Android ARM device in practice. */
    private const val CLK_TCK = 100L

    data class Snapshot(
        /** App CPU time, utime+stime, milliseconds. Sums across all threads. */
        val cpuTimeMs: Long,
        val pssMb: Long,
        val nativeHeapMb: Long,
        /** System-wide MemAvailable. Headroom before the LMK starts killing. */
        val availMemMb: Long,
        /** Hottest relevant thermal zone, celsius. -1 if none readable. */
        val thermalC: Double,
        /** Highest current core frequency, MHz. -1 if unreadable. */
        val cpuFreqMhz: Int,
        val uptimeMs: Long,
    )

    // ── /proc/self/stat ──────────────────────────────────────────────────────

    /**
     * Fields 14 (utime) and 15 (stime), 1-indexed. Parsed after the last ')'
     * because field 2 is the process name and may itself contain spaces and
     * parentheses — splitting the whole line on whitespace is a classic bug.
     */
    private fun appCpuTimeMs(): Long = try {
        val stat = File("/proc/self/stat").readText()
        val after = stat.substring(stat.lastIndexOf(')') + 2)
        val f = after.split(' ')
        // after[0] is field 3, so utime is index 11 and stime index 12
        val utime = f[11].toLong()
        val stime = f[12].toLong()
        (utime + stime) * 1000L / CLK_TCK
    } catch (e: Exception) {
        -1L
    }

    // ── thermal ──────────────────────────────────────────────────────────────

    /**
     * Zones whose type looks like the SoC or CPU. Enumerated once: MediaTek
     * exposes 50+ zones and reading all of them every sample is wasteful.
     */
    private val thermalPaths: List<File> by lazy {
        val wanted = listOf("cpu", "soc", "tsens", "mtktscpu", "ap", "big", "little")
        val found = ArrayList<File>()
        try {
            File("/sys/class/thermal").listFiles()
                ?.filter { it.name.startsWith("thermal_zone") }
                ?.forEach { zone ->
                    val type = runCatching {
                        File(zone, "type").readText().trim().lowercase()
                    }.getOrNull() ?: return@forEach
                    val temp = File(zone, "temp")
                    if (temp.canRead() && wanted.any { type.contains(it) }) {
                        found.add(temp)
                    }
                }
        } catch (_: Exception) { }

        if (found.isEmpty()) {
            // No zone matched by name; fall back to every readable zone and let
            // the max speak. Better a coarse number than none.
            try {
                File("/sys/class/thermal").listFiles()
                    ?.filter { it.name.startsWith("thermal_zone") }
                    ?.map { File(it, "temp") }
                    ?.filter { it.canRead() }
                    ?.let { found.addAll(it) }
            } catch (_: Exception) { }
        }
        Log.i(TAG, "thermal zones tracked: ${found.size}")
        found
    }

    /** Zones report millidegrees, decidegrees or degrees depending on vendor. */
    private fun normaliseTemp(raw: Long): Double = when {
        raw > 1000 -> raw / 1000.0
        raw > 200 -> raw / 10.0
        else -> raw.toDouble()
    }

    private fun thermalC(): Double {
        var hottest = -1.0
        for (f in thermalPaths) {
            val v = runCatching { f.readText().trim().toLong() }.getOrNull() ?: continue
            val c = normaliseTemp(v)
            if (c in 1.0..150.0 && c > hottest) hottest = c
        }
        return hottest
    }

    // ── cpu frequency ────────────────────────────────────────────────────────

    private val freqPaths: List<File> by lazy {
        (0 until 8).map {
            File("/sys/devices/system/cpu/cpu$it/cpufreq/scaling_cur_freq")
        }.filter { it.canRead() }
    }

    /**
     * Highest current core frequency. A sustained fall across a run is direct
     * evidence of thermal throttling, and unlike temperature it is almost
     * always readable.
     */
    private fun cpuFreqMhz(): Int {
        var top = -1
        for (f in freqPaths) {
            val khz = runCatching { f.readText().trim().toInt() }.getOrNull() ?: continue
            val mhz = khz / 1000
            if (mhz > top) top = mhz
        }
        return top
    }

    // ── memory ───────────────────────────────────────────────────────────────

    private fun availMemMb(): Long = try {
        File("/proc/meminfo").useLines { lines ->
            lines.firstOrNull { it.startsWith("MemAvailable:") }
                ?.filter { it.isDigit() }?.toLong()?.div(1024) ?: -1L
        }
    } catch (e: Exception) {
        -1L
    }

    // ── snapshot ─────────────────────────────────────────────────────────────

    /**
     * PSS only, for per-token sampling.
     *
     * snapshot() reads /proc/self/stat, several thermal zones, and every cpufreq
     * node. That is fine a few times per question, but calling it once per generated
     * token would add measurable overhead to the thing being measured. This is the
     * single binder call and nothing else.
     */
    fun pssMb(): Long {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024L
    }

    fun snapshot(): Snapshot {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return Snapshot(
            cpuTimeMs = appCpuTimeMs(),
            pssMb = info.totalPss / 1024L,
            nativeHeapMb = Debug.getNativeHeapAllocatedSize() / (1024L * 1024L),
            availMemMb = availMemMb(),
            thermalC = thermalC(),
            cpuFreqMhz = cpuFreqMhz(),
            uptimeMs = android.os.SystemClock.elapsedRealtime(),
        )
    }

    /**
     * Accumulates samples taken during one question, so peaks and means are
     * available rather than only the endpoints. Sampling is cheap — a handful
     * of small sysfs reads — but call it at a throttled rate, not per token.
     */
    class Session(private val start: Snapshot) {
        private var peakPss = start.pssMb
        private var peakThermal = start.thermalC
        private var minFreq = start.cpuFreqMhz
        private var maxFreq = start.cpuFreqMhz
        private var freqSum = 0L
        private var freqN = 0
        private var minAvailMem = start.availMemMb

        fun sample() {
            val s = snapshot()
            if (s.pssMb > peakPss) peakPss = s.pssMb
            if (s.thermalC > peakThermal) peakThermal = s.thermalC
            if (s.cpuFreqMhz > 0) {
                if (minFreq <= 0 || s.cpuFreqMhz < minFreq) minFreq = s.cpuFreqMhz
                if (s.cpuFreqMhz > maxFreq) maxFreq = s.cpuFreqMhz
                freqSum += s.cpuFreqMhz
                freqN++
            }
            if (s.availMemMb in 0 until minAvailMem) minAvailMem = s.availMemMb
        }

        /** Flat map ready to merge into an EvalRunner record. */
        fun finish(): Map<String, Any?> {
            val end = snapshot()
            val wallMs = end.uptimeMs - start.uptimeMs
            val cpuMs = if (start.cpuTimeMs >= 0 && end.cpuTimeMs >= 0)
                end.cpuTimeMs - start.cpuTimeMs else -1L

            return mapOf(
                // CPU time is the headline: unaffected by cooldown or by
                // background load.
                "cpu_ms" to cpuMs,
                "cpu_util_pct" to if (cpuMs >= 0 && wallMs > 0)
                    100.0 * cpuMs / wallMs else -1.0,
                "wall_ms" to wallMs,

                "pss_mb_start" to start.pssMb,
                "pss_mb_peak" to peakPss,
                "pss_mb_end" to end.pssMb,
                "native_heap_mb_end" to end.nativeHeapMb,
                "avail_mem_mb_min" to minAvailMem,

                "thermal_c_start" to start.thermalC,
                "thermal_c_peak" to peakThermal,
                "thermal_c_end" to end.thermalC,

                "cpu_freq_mhz_min" to minFreq,
                "cpu_freq_mhz_max" to maxFreq,
                "cpu_freq_mhz_mean" to if (freqN > 0) freqSum.toDouble() / freqN else -1.0,
            )
        }
    }

    fun begin(): Session = Session(snapshot())

    /** One-off probe, logged at startup so an unreadable sensor is visible
     *  immediately rather than as a column of -1 after a 45-minute run. */
    fun logCapabilities() {
        val s = snapshot()
        Log.i(TAG, "cpu_time=${s.cpuTimeMs}ms  thermal=${s.thermalC}C " +
                "(${thermalPaths.size} zones)  freq=${s.cpuFreqMhz}MHz " +
                "(${freqPaths.size} cores)  avail=${s.availMemMb}MB")
        if (s.thermalC < 0) Log.w(TAG, "no thermal zone readable")
        if (s.cpuFreqMhz < 0) Log.w(TAG, "cpufreq not readable")
    }
}
