import api from './client.ts'


export type PassageModel = {
 reference: string
}

export type PassageResponse = {
    study_notes: string
}

export function apiPassage(reference: string): Promise<PassageResponse> {
    return api.get("/study_notes", { params : {reference: reference} }).then(response => response.data)
}