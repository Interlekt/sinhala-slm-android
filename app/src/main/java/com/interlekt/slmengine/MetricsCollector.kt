package com.interlekt.slmengine

import android.os.Debug
import java.io.File

object MetricsCollector {

    fun ramUsedMB(): Long {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024L
    }

    private var lastTotal = 0L
    private var lastIdle  = 0L

    fun cpuPercent(): Float {
        return try {
            val line = File("/proc/stat").readLines().first()
            val vals = line.trim().split("\\s+".toRegex()).drop(1).map { it.toLong() }
            val idle  = vals[3]
            val total = vals.sum()
            val pct   = if (total - lastTotal == 0L) 0f
            else 100f * (1f - (idle - lastIdle).toFloat() /
                    (total - lastTotal).toFloat())
            lastTotal = total
            lastIdle  = idle
            pct.coerceIn(0f, 100f)
        } catch (e: Exception) { 0f }
    }

    fun msPerToken(elapsedMs: Long, tokenCount: Int): Float =
        if (tokenCount == 0) 0f else elapsedMs / tokenCount.toFloat()
}
