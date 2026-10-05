package kr.sonkkeut.android

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONObject

/** Stores menu documents only. Audio and user transcripts are never persisted. */
class MenuRagDatabase(context: Context) : SQLiteOpenHelper(context, "sonkkeut_menu_rag.db", null, 1) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE catalog(scope TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    }
    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    @Synchronized fun catalog(scope: String, payload: String): String {
        require(scope.isNotBlank() && scope.length <= 512) { "매장 문맥이 올바르지 않습니다." }
        require(payload.length <= 1000000) { "메뉴 데이터가 너무 큽니다." }
        val input = JSONArray(payload)
        require(input.length() <= 1000) { "메뉴는 1,000개 이내여야 합니다." }
        val documents = JSONArray()
        val names = mutableSetOf<String>()
        for (index in 0 until input.length()) {
            val item = input.getJSONObject(index)
            val name = item.getString("name")
            require(name.isNotBlank() && name.length <= 80 && names.add(name)) { "메뉴 이름이 올바르지 않습니다." }
            val aliases = item.getJSONArray("aliases")
            require(aliases.length() <= 100) { "메뉴 별칭이 너무 많습니다." }
            for (i in 0 until aliases.length()) { require(aliases.getString(i).length in 1..80) }
            documents.put(JSONObject().put("name", name).put("aliases", aliases)
                .put("sold_out", item.getBoolean("sold_out")).put("category", item.optString("category", "")))
        }
        val canonical = documents.toString()
        val db = writableDatabase
        db.beginTransaction()
        try {
            val existing = db.rawQuery("SELECT payload FROM catalog WHERE scope = ?", arrayOf(scope)).use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
            if (existing != canonical) {
                db.execSQL("INSERT OR REPLACE INTO catalog(scope,payload) VALUES (?,?)", arrayOf(scope, canonical))
            }
            // Read the authoritative, scoped DB document snapshot used for retrieval.
            val stored = db.rawQuery("SELECT payload FROM catalog WHERE scope = ?", arrayOf(scope)).use { cursor ->
                check(cursor.moveToFirst()); cursor.getString(0)
            }
            db.setTransactionSuccessful()
            return stored
        } finally { db.endTransaction() }
    }
}
