// automatically generated by the FlatBuffers compiler, do not modify

import com.google.flatbuffers.BaseVector
import com.google.flatbuffers.BooleanVector
import com.google.flatbuffers.ByteVector
import com.google.flatbuffers.Constants
import com.google.flatbuffers.DoubleVector
import com.google.flatbuffers.FlatBufferBuilder
import com.google.flatbuffers.FloatVector
import com.google.flatbuffers.LongVector
import com.google.flatbuffers.StringVector
import com.google.flatbuffers.Struct
import com.google.flatbuffers.Table
import com.google.flatbuffers.UnionVector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sign

@Suppress("unused")
@kotlin.ExperimentalUnsignedTypes
class Rapunzel : Struct() {

    fun __init(_i: Int, _bb: ByteBuffer)  {
        __reset(_i, _bb)
    }
    fun __assign(_i: Int, _bb: ByteBuffer) : Rapunzel {
        __init(_i, _bb)
        return this
    }
    val hairLength : Int get() = bb.getInt(bb_pos + 0)
    fun mutateHairLength(hairLength: Int) : ByteBuffer = bb.putInt(bb_pos + 0, hairLength)
    companion object {
        fun createRapunzel(builder: FlatBufferBuilder, hairLength: Int) : Int {
            builder.prep(4, 4)
            builder.putInt(hairLength)
            return builder.offset()
        }
    }
}
