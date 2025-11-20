import api from './client.ts'


export type PassageModel = {
 reference: string
}

export function apiPassage(reference: string): Promise<string> {
    return api.get("/study_notes", { params : {reference: reference} })
}